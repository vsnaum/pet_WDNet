from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms

class Getdata(torch.utils.data.Dataset):
	def __init__(self,dataset_dir,train=True):
		self.train = train
		self.transform_norm=transforms.Compose([transforms.ToTensor()])
		self.transform_tensor= transforms.ToTensor()
		#root = './dataset/CLWD/train/' 
		if self.train:
			root = dataset_dir + '/train/'
		else:
			root = dataset_dir + '/test/'
		self.imageJ_path=osp.join(root,'Watermarked_image','%s.jpg')
		self.imageI_path=osp.join(root,'Watermark_free_image','%s.jpg')
		self.mask_path=osp.join(root,'Mask','%s.png')
		if self.train:
			self.balance_path=osp.join(root,'Loss_balance','%s.png')
			self.alpha_path=osp.join(root,'Alpha','%s.png')
			self.W_path=osp.join(root,'Watermark','%s.png')
		self.root = root
		self.transform= transforms
		self.ids = list()
		for file in os.listdir(root+'Watermarked_image'):
			#if(file[:-4]=='.jpg'):
			self.ids.append(file.strip('.jpg'))
	def __getitem__(self,index):
		if self.train:
			imag_J,image_I,mask,balance,alpha,w=self.pull_item(index)
			return imag_J,image_I,mask,balance,alpha,w
		else:
			imag_J,image_I,mask=self.pull_item(index)
			return imag_J,image_I,mask
	def __len__(self):
		return len(self.ids)
	def pull_item(self,index):
		img_id = self.ids[index]
		img_J=Image.open(self.imageJ_path%img_id)
		img_I=Image.open(self.imageI_path%img_id)
		mask = Image.open(self.mask_path%img_id)
		if self.train:
			balance = Image.open(self.balance_path%img_id)
			alpha = Image.open(self.alpha_path%img_id)
			w = Image.open(self.W_path%img_id)

		img_source = self.transform_norm(img_J) # Watermarked_image
		image_target = self.transform_norm(img_I) # Watermark_free_image
		mask=self.transform_tensor(mask)
		if self.train:
			w=self.transform_norm(w)
			alpha=self.transform_tensor(alpha)
			balance = self.transform_tensor(balance)
			return img_source,image_target,mask,balance,alpha,w
		else:
			return img_source,image_target,mask
		