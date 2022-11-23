import os
import sys
import easydict


def get_config():
    args = easydict.EasyDict(
        {
            'arch': 'pmg',
            'model_path': '../sofar_v4_pmg_pre_trained=imagenet_lb=0.05.pth',                              # To be changed 

            'dataset': 'sofar_v3',
            'data_root_path': '../data_folder/',   # To be changed 

            'train_class_name': 'outer_normal,outer_damage,outer_dirt,outer_wash,inner_wash,inner_dashboard,inner_cupholder,inner_glovebox,inner_washer_fluid,inner_rear_seat,inner_sheet_dirt', 
            'test_class_name': 'outer_normal,outer_damage,outer_dirt,outer_wash,inner_wash,inner_dashboard,inner_cupholder,inner_glovebox,inner_washer_fluid,inner_rear_seat,inner_sheet_dirt', 

            'num_workers': 4, 
            'batch_size': 1 ,

            'save_img': True, # To show result images
            'save_path': '../results/analysis/'
        }
    )
    args.pretext_task = 'none'
    
    return args

