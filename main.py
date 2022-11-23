import os
import sys 
import shutil
import torch 
import easydict
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw

from downstream_modules.data_utils import create_dataloader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from config import get_config

from utils import *
from model.set_model import set_model
from dataset import load_dataloaders
from sklearn.metrics import *
from downstream_modules.utils import calculate_metrics


def main():
    
    # configuration
    args = get_config()
    
    # load dataset & model
    args, query_dataset, query_loader, support_loaders = load_dataloaders(args)
    model = load_model(args)
    
    # assign saved location
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # start retrieval
    for idx, support_loader in enumerate(support_loaders) :
        support_set_features = get_support_vectors(support_loader, model)
        sorted_dict = retriever(model, query_dataset, query_loader, support_set_features)
    
        # save the result
        for file_idx, file_path in enumerate(sorted_dict) :
            dst_path = f'./result/{idx}_sorted/' + str(file_idx) + '_' + file_path[0].split('/')[-1]
            shutil.copy(file_path[0], dst_path)
        
        # remove vars
        del support_set_features
        del sorted_dict

        
        
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    



