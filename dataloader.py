from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from get_data import Getdata
def dataloader(dataset, batch_size, num_workers=0):
	data_loader=DataLoader(
            Getdata(dataset),
            batch_size=batch_size, shuffle=True, num_workers=num_workers)
	return data_loader