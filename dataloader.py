from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from WDNet_main.get_data import Getdata
def dataloader(dataset, input_size, batch_size):
	data_loader=DataLoader(
            Getdata(dataset),
            batch_size=batch_size, shuffle=True, num_workers=6)
	return data_loader