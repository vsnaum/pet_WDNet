import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader, dataloader_test
from unet_parts import *
from tensorboardX import SummaryWriter
from vgg import Vgg16
import pytorch_ssim
import my_helpers as mh
     


def calc_iou(prediction, ground_truth):
    prediction, ground_truth = prediction.to('cpu'), ground_truth.to('cpu')
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += np.logical_and(prediction[i].detach().numpy() > 0, ground_truth[i].detach().numpy() > 0).astype(np.float32).sum()
        union += np.logical_or(prediction[i].detach().numpy() > 0, ground_truth[i].detach().numpy() > 0).astype(np.float32).sum()
    return float(intersection) / union


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dilation=nn.Sequential(
          nn.Conv2d(512,512,3,1,2, dilation=2),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512,512,3,1,4, dilation=4),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512,512,3,1,6, dilation=6),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          )
        self.outw = OutConv(64, 3)
        self.outa = OutConv(64, 1)
        self.out_mask = OutConv(64, 1)
        self.sg=nn.Sigmoid()
        self.other=OutConv(64, 64)
        self.post_process_1=nn.Sequential(
          nn.Conv2d(64+6, 64, 3, 1, 1),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.2),
          nn.Conv2d(64, 128, 3, 1, 1),
          )
        self.post_process_2=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_3=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_4=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_5=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 3, 3, 1, 1),
          nn.Sigmoid(),
          )
    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dilation(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        w = self.outw(x)
        a = self.outa(x)
        other = self.other(x)
        other = self.sg(other)
        mask = self.out_mask(x)
        mask=self.sg(mask)
        a=self.sg(a)
        w=self.sg(w)
        a=mask*a
        I_watermark=(x0-a*w)/(1.0-a+1e-6)
        I_watermark=torch.clamp(I_watermark,0,1)
        xx1=self.post_process_1(torch.cat([other,I_watermark,x0],1))
        xx2=self.post_process_2(xx1)
        xx3=self.post_process_3(xx1+xx2)
        xx4=self.post_process_4(xx2+xx3)
        I_watermark2=self.post_process_5(xx4+xx3)
        I=I_watermark2*mask+(1.0-mask)*x0
        return I,mask,a,w,I_watermark

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=3, output_dim=1):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        #utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        return x

class WDNet(object):
    def __init__(self, args):
        # parameters
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.save_dir = args['save_dir']
        self.dataset = args['dataset']
        self.gpu_mode = args['gpu_mode']
        self.dataloader_workers = args['dataloader_workers']
        self.train_hist = []
        self.z_dim = 62
        self.class_num = 3
        self.sample_num = self.class_num ** 2

        # load dataset
        self.data_loader = dataloader(self.dataset, self.batch_size, self.dataloader_workers)
        self.data_loader_test = dataloader_test(self.dataset, self.batch_size, self.dataloader_workers)

        def weight_init(m):
          if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
          elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)
        
        # networks init
        self.G = generator(3, 3)
        self.D = discriminator(input_dim=6, output_dim=1)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args['lrG'], betas=(args['beta1'], args['beta2']))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args['lrD'], betas=(args['beta1'], args['beta2']))
        
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.l1loss=nn.L1Loss().cuda()
            self.loss_mse = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.loss_mse = nn.MSELoss()
            self.l1loss=nn.L1Loss()
        self.G.apply(weight_init)
        self.D.apply(weight_init)

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

        print('Model initiated. Do not forget load state dicts if you need pretrained model')

    def train(self, prev_train_hist=False):
        if prev_train_hist:
            self.train_hist = mh.pkl_load(os.path.join(self.save_dir,prev_train_hist))


        vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.D.train()
        print(f'{mh.curr_time()} Start training!')


        lenth=self.data_loader.dataset.__len__()
        if prev_train_hist:
            iter_all=len(self.train_hist[-1]['iter_all'])
        else:
            iter_all = 0

        for epoch in range(self.epoch):
            self.G.train()

            for iter, (x_, y_, mask, balance, alpha, w) in enumerate(self.data_loader):
                iter_all+=1 #iter+epoch*(lenth//self.batch_size)
                if iter ==  lenth // self.batch_size:
                    break

                if self.gpu_mode:
                    x_, y_, mask, balance, alpha, w=x_.cuda(), y_.cuda(), mask.cuda(), balance.cuda(), alpha.cuda(), w.cuda()

                # update D network
                if ((iter+1)%3)==0 :
                  self.D_optimizer.zero_grad()
  
                  D_real = self.D(x_, y_)
                  D_real_loss = self.BCE_loss(D_real, torch.ones_like(D_real))
  
                  G_ ,g_mask, g_alpha, g_w,I_watermark= self.G(x_)
                  D_fake = self.D(x_,G_)
                  D_fake_loss = self.BCE_loss(D_fake, torch.zeros_like(D_fake))
  
                  D_loss = 0.5*D_real_loss + 0.5*D_fake_loss
                  D_loss.backward()
                  self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ ,g_mask, g_alpha, g_w,I_watermark= self.G(x_)
                D_fake = self.D(x_, G_)
                G_loss = self.BCE_loss(D_fake, torch.ones_like(D_fake))
                feature_G = vgg(G_)
                feature_real=vgg(y_)
                vgg_loss=0.0
                for j in range (3):
                  vgg_loss+= self.loss_mse(feature_G[j],feature_real[j])

                mask_loss=self.l1loss(g_mask*balance,mask*balance)*balance.size(0)*balance.size(1)*balance.size(2)*balance.size(3)/balance.sum()
                w_loss=self.l1loss(g_w*mask,w*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                alpha_loss=self.l1loss(g_alpha*mask,alpha*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                I_watermark_loss=self.l1loss(I_watermark*mask,y_*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                I_watermark2_loss=self.l1loss(G_*mask,y_*mask)*mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)/mask.sum()
                G_writer=G_loss.data
                G_loss=G_loss+10.0*mask_loss+10.0*w_loss+10.0*alpha_loss+50.0*(0.7*I_watermark2_loss+0.3*I_watermark_loss)+1e-2*vgg_loss
                G_loss.backward()
                self.G_optimizer.step()

                if ((iter+1)%100) == 0:
                    time_stamp = mh.curr_time()

                    self.train_hist.append({})
                    self.train_hist[-1]['time_stamp'] = time_stamp
                    self.train_hist[-1]['epoch'] = epoch + 1
                    self.train_hist[-1]['iter_all'] = iter_all
                    self.train_hist[-1]['G_Loss'] = G_writer
                    self.train_hist[-1]['D_Loss'] = D_loss.item()
                    self.train_hist[-1]['W_Loss'] = w_loss
                    self.train_hist[-1]['alpha_Loss'] = alpha_loss
                    self.train_hist[-1]['mask_Loss'] = mask_loss
                    self.train_hist[-1]['I_watermark_Loss'] = I_watermark_loss
                    self.train_hist[-1]['I_watermark2_Loss'] = I_watermark2_loss
                    self.train_hist[-1]['vgg_Loss'] = vgg_loss

                    
                
                    log_text = f"{time_stamp}   Epoch: [{(epoch + 1):2d}] [{(iter + 1):4d}/{self.data_loader.dataset.__len__() // self.batch_size:4d}] D_loss: {D_loss.item():8f}, G_loss: {G_writer:8f}" 
                    print(log_text)
                    self.save(prefix = time_stamp)

        print(f"{mh.curr_time()} Training finished!")

    def test(self,verbose=False):
        self.G.eval()
        if self.gpu_mode:
            self.G.cuda()

        self.test_results = []
        self.ssim = pytorch_ssim.SSIM()
        mask_bce = 0.0
        img_bce = 0.0
        mask_mse = 0.0
        img_mse = 0.0
        img_l1 = 0.0
        img_ssim = 0.0
        mask_ssim= 0.0
        with torch.no_grad():
            for iter, (x_, y_, mask) in enumerate(self.data_loader_test):
                if self.gpu_mode:
                    x_ = x_.cuda()
                    y_ = y_.cuda()
                    mask = mask.cuda()
                G_ ,g_mask, _, _,_= self.G(x_)
                mask_pred = (g_mask > (15.0/255.0)).float()
                mask_truth = mask[:,1,:,:].unsqueeze(1)

                mask_bce += self.BCE_loss(mask_pred,mask_truth).item()
                mask_mse += self.loss_mse(mask_pred,mask_truth).item()
                mask_ssim += self.ssim(mask_pred,mask_truth).item()
                
                img_bce += self.BCE_loss(G_,y_).item()
                img_mse += self.loss_mse(G_,y_).item()
                img_l1 += self.l1loss(G_,y_).item()
                img_ssim += self.ssim(G_,y_).item()

                iter_result = {'iter':iter+1, 'mask_BCE': mask_bce/(iter+1), 'mask_MSE': mask_mse/(iter+1), 'mask_SSIM': mask_ssim/(iter+1),
                               'img_BCE': img_bce/(iter+1), 'img_MSE': img_mse/(iter+1), 'img_L1': img_l1/(iter+1), 'img_SSIM': img_ssim/(iter+1)}

                self.test_results.append(iter_result)
                if verbose:
                    print(f"iter {iter_result['iter']:4d}     mask_BCE {iter_result['mask_BCE']:5.4f}   mask_MSE {iter_result['mask_MSE']:5.4f}   mask_SSIM {iter_result['mask_SSIM']:5.4f}     img_BCE {iter_result['img_BCE']:5.4f}   img_MSE {iter_result['img_MSE']:5.4f}   img_L1 {iter_result['img_L1']:5.4f}   img_SSIM {iter_result['img_SSIM']:5.4f}")
        
        print(f"Results:    mask_BCE {iter_result['mask_BCE']:5.4f}    mask_BCE {iter_result['mask_BCE']:5.4f}   mask_MSE {iter_result['mask_MSE']:5.4f}   mask_SSIM {iter_result['mask_SSIM']:5.4f}     img_BCE {iter_result['img_BCE']:5.4f}   img_MSE {iter_result['img_MSE']:5.4f}   img_L1 {iter_result['img_L1']:5.4f}   img_SSIM {iter_result['img_SSIM']:5.4f}")
        
        mh.pkl_dump(os.path.join(self.save_dir,'WDNet_test.pkl'),self.test_results)
        print(f'Test data saved to {self.save_dir}')

            
        

    def save(self,prefix=False):
        if prefix:
            torch.save(self.G.state_dict(), os.path.join(self.save_dir,f'{prefix}__WDNet_G.sd'))
            torch.save(self.D.state_dict(), os.path.join(self.save_dir,f'{prefix}__WDNet_D.sd'))
            mh.pkl_dump(os.path.join(self.save_dir,f'{prefix}__train_hist.pkl'),self.train_hist)
        else:
            torch.save(self.G.state_dict(), os.path.join(self.save_dir,f'WDNet_G.sd'))
            torch.save(self.D.state_dict(), os.path.join(self.save_dir,f'WDNet_D.sd'))
            mh.pkl_dump(os.path.join(self.save_dir,f'train_hist.pkl'),self.train_hist)
        print(f'Model saved to {self.save_dir}, prefix={prefix}')

    def load(self,prefix=False, load_train_hist=False):
        if prefix:
            self.G.load_state_dict(torch.load(os.path.join(self.save_dir,f'{prefix}__WDNet_G.sd')))
            self.D.load_state_dict(torch.load(os.path.join(self.save_dir,f'{prefix}__WDNet_D.sd')))
        else:
            self.G.load_state_dict(torch.load(os.path.join(self.save_dir,f'WDNet_G.sd')))
            self.D.load_state_dict(torch.load(os.path.join(self.save_dir,f'WDNet_D.sd')))
        print(f'Model loaded from {self.save_dir}, prefix={prefix}')
