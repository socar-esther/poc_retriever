import os
import sys 
import torch 
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw

from model.set_model import set_model
from sklearn.metrics import *
from downstream_modules.utils import calculate_metrics

def load_model(args):
    model = set_model(args)
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    print('pre-trained model is loaded')
    return model

def save_compared_imgs(args, imgs, labels, preds, idx): 
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1).repeat(1,448,448)    
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1).repeat(1,448,448)

    class_to_name = [k for k in list(name_to_class.keys())]
    
    count = 1 
    for (img, label, pred) in zip(imgs, labels, preds):
        # wrong pred인 경우에만 저장 
        if pred != label:
            img = img * std + mean 

            img = img.numpy() * 255
            img = Image.fromarray(np.transpose(img, (1, 2, 0)).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            text = 'label:{} || pred: {} '.format(class_to_name[label], class_to_name[pred])
            textwidth, textheight = draw.textsize(text)

            width, height = img.size 
            margin = 10
            x = width - textwidth - margin
            y = height - textheight - margin
            

            draw.text((x, y), text)

            file_path = os.path.join(args.save_path, str(idx * args.batch_size + count) + '.jpg')
            img.save(file_path)

            count += 1
            
def get_support_vectors(support_loader, model):
    for idx, (img, target) in enumerate(support_loader) : 
        img = img.cuda()
        target = target.cuda()

        out1, out2, out3, out_concat = model._forward(img)
        print(out_concat.shape)
        if idx == 0:
            support_set_features = out_concat
        else:
            support_set_features = np.concat((support_set_features, out_concat), axis = 1)
        break
        
    return support_set_features


def retrieve(model, query_dataset, query_loader, support_set_features):
    result_dict = dict()
    model.eval()
    distance_opt = 'euclidean'

    for idx, (img, target) in enumerate(query_loader) : 
        img = img.cuda()
        target = target.cuda()
        img_path = query_dataset.imgs[idx][0]

        query_vec = model._forward(img)
        query_vec = query_vec#.cpu().detach().numpy()
        query_vec = query_vec[0].cpu().detach().numpy()

        distance_list = list()
        for i in range(len(support_set_features)) :
            support_vec = support_set_features[i].reshape(1, -1)
            support_vec = support_vec#.cpu().detach().numpy()
            if distance_opt == 'euclidean' :
                support_vec = support_vec.cpu().detach().numpy()
                dist = np.linalg.norm(support_vec - query_vec) # euclidean distacne
            elif distance_opt == 'cosine' :
                dist = distance.cosine(support_vec, query_vec) # cosine distance
            else :
                raise NotImplementedError()

            distance_list.append(dist)
        avg_dist = np.mean(distance_list)
        result_dict[img_path] = avg_dist
        
    # for Log
    #sorted_dict = sorted(result_dict.items(), key = lambda item: item[1])
    #print(sorted_dict)
    
    #print('check result_dict:', result_dict)
    return result_dict
