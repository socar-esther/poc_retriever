import torch 
import torch.nn as nn 
import torchvision

def set_model(args):
    if args.arch == 'resnet':
        if args.pretext_task == 'imagenet':
            model = torchvision.models.resnet50(pretrained=True)
            in_feats = model.fc.in_features
        else:
            model = torchvision.models.resnet50(pretrained=False)
            in_feats = model.fc.in_features
            model.fc = nn.Identity()
            if args.pretext_task == 'byol':
                model = load_model(model, args.pretext_task, args.pretrained_path)
            
        model.fc = nn.Sequential(
            nn.Linear(
                in_feats,
                len(args.train_class) 
        ))

    elif args.arch == 'pmg':
        from model.resnet import resnet50
        from model.pmg import PMG
        
        if args.pretext_task == 'imagenet':
            model = resnet50(pretrained=True)
        else:
            model = resnet50(pretrained=False)
            model.fc = nn.Identity()
            if args.pretext_task == 'byol':
                model = load_model(model, args.pretext_task, args.pretrained_path)
        
        #print('> check the lengths of classes:', len(args.test_class_name))
        model = PMG(model, feature_size = 512, num_classes = len(args.test_class_name))
    
    return model 

def load_model(model, pretext_task, path, net='online'):
    if pretext_task == 'byol':
        state_dict = torch.load(path)['state_dict']

        new_state_dict = {}
        for key in state_dict:
            # online model
            if net == 'online':
                if 'encoder' in key and 'momentum' not in key:
                    new_key = key[len('encoder.'):]
                    new_state_dict[new_key] = state_dict[key]
                    print(key, '-->', new_key)
            # target model 
            else:
                if 'encoder' in key and 'momentum' in key:
                    new_key = key[len('momentum.encoder.'):]
                    new_state_dict[new_key] = state_dict[key]
                    print(key, '-->', new_key)
                                    
        model.load_state_dict(new_state_dict, strict=True)
        print('Successfully Loaded the Weight!') 
    return model 