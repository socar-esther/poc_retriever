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


class retriever:
    
    def __init__(self, args) :
        # load dataset & model
        self.args, self.query_dataset, self.query_loader, self.support_loaders = load_dataloaders(args)
        self.model = load_model(self.args)
    
            
    def inference(self) :
        for idx, support_loader in enumerate(self.support_loaders) :
            # start retrieve
            support_set_features = get_support_vectors(support_loader, self.model)
            sorted_dict = retrieve(self.model, self.query_dataset, self.query_loader, support_set_features)

            # save the result
            tmp_sorted_order_list = list()
            
            for file_idx, file_path in enumerate(sorted_dict) :
                tmp_sorted_order_list.append(file_path[0])

            del support_set_features
            del sorted_dict
            
            if idx == 0 :
                sorted_08_order_list = tmp_sorted_order_list.copy()
                
            elif idx == 1:
                sorted_16_order_list = tmp_sorted_order_list.copy()
                
        
        return sorted_08_order_list, sorted_16_order_list
        
        
if __name__ == '__main__' :
    args = get_config()
    retriever = retriever(args)
    sorted_08_order_list, sorted_16_order_list = retriever.inference() # want to add this! 
    
    #print('sorted class 08 Top10:', sorted_08_order_list[:10])
    #print('sorted class 16 Top10:', sorted_16_order_list[:10])
    