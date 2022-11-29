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
            result_dict = retrieve(self.model, self.query_dataset, self.query_loader, support_set_features)

            # save the result
            tmp_sorted_dist_list = list()
            
            for file_nm, distance in result_dict.items() :
                tmp_sorted_dist_list.append(distance)

            del support_set_features
            del result_dict
            
            if idx == 0 :
                sorted_08_dist_list = tmp_sorted_dist_list.copy()
                
            elif idx == 1:
                sorted_16_dist_list = tmp_sorted_dist_list.copy()
                
        
        return sorted_08_dist_list, sorted_16_dist_list
        
        
if __name__ == '__main__' :
    args = get_config()
    retriever = retriever(args)
    sorted_08_dist_list, sorted_16_dist_list = retriever.inference() # want to add this! 
    
    #print('distance with class 08 data:', sorted_08_dist_list[:10]) 
    #print('distance with class 16 data:', sorted_16_dist_list[:10])
    
