import os
import torch 
import numpy as np

from downstream_modules.data_utils import create_dataloader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

def load_dataloaders(args):
    normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    transform_options = transforms.Compose([
                        transforms.Resize((550, 550)),
                        transforms.RandomCrop(448, padding=8),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                    ])
    
    # (1) load test(=query) dataset
    query_dataset = ImageFolder(root='../data_folder/query_set', transform=transform_options)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=1, shuffle=True)
    
    # (2) load support(=sample) dataset
    support_08_dataset = ImageFolder(root='../data_folder/support_set/08_inner_cupholder_dirt', transform=transform_options)
    support_08_loader = torch.utils.data.DataLoader(support_08_dataset, batch_size=32, shuffle=True)

    # support_14_dataset = ImageFolder(root='../data_folder/support_set/14_inner_sheet', transform=transform_options)
    # support_14_loader = torch.utils.data.DataLoader(support_14_dataset, batch_size=128, shuffle=True)

    support_16_dataset = ImageFolder(root='../data_folder/support_set/16_inner_seat_dirt', transform=transform_options)
    support_16_loader = torch.utils.data.DataLoader(support_16_dataset, batch_size=32, shuffle=True)
    
    # assign output class's name
    args.test_class_name = ['outer_normal','outer_damage','outer_dirt','outer_wash','inner_wash','inner_dashboard','inner_cupholder','inner_cupholder_dirt','inner_glovebox','inner_washer_fluid','inner_front_seat','inner_rear_seat','inner_trunk','inner_sheet_dirt']
    
    support_loaders = [support_08_loader, support_16_loader]
    
    return args, query_dataset, query_loader, support_loaders