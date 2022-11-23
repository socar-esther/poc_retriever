import os 
import random 
import pandas as pd 
from PIL import Image
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def create_dataloader(args):
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    
    if args.dataset == 'sofar_v4':
        data_path = os.path.join(args.data_root_path, "SOFAR-Image-v4-Split")
        train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        _name_to_class = train_dataset.find_classes(os.path.join(data_path, 'train'))[1] 
        test_dataset = ImageFolder(os.path.join(data_path, 'test'), transform=test_transform)
    else:
        data_path = args.data_root_path #os.path.join(args.data_root_path, 'sofar_dataset_{}'.format(args.dataset.split('_')[-1]))

        train_dataset = ImageFolder(data_path, 
                                        transform=train_transform,
                                    )
        #_name_to_class = train_dataset.find_classes(os.path.join(data_path, 'train'))[1] 
        
        test_dataset = ImageFolder(data_path, 
                                    transform=test_transform)
    
    name_to_class = {}
    #for k, v in _name_to_class.items():
    #    name_to_class[k[3:]] = v
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )        
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )      
    
    return train_loader, test_loader#, name_to_class

def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                           y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size,
                y * block_size:(y + 1) * block_size] = temp

    return jigsaws