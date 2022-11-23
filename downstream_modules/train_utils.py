import os
import wandb 
import numpy as np
from tqdm import tqdm 
from sklearn.metrics import *

import torch 
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .utils import EarlyStopping, calculate_metrics
from .data_utils import jigsaw_generator

def train(model, train_loader, test_loader, device, args):
    # model 
    model.to(device)
    
    # optimizer 
    optimizer, scheduler = load_optimizer(model, args.arch, args.lr, args.lr_decay, args.weight_decay, args.epochs, args.momentum)

    # for logging
    wandb.init(project="car_state_classifier", name= args.arch + '_' + args.exp_name)
    model_save_base = os.path.join('./artifacts', args.dataset, args.arch + '_pre-trained=' + args.pretext_task, args.exp_name)    
    if not os.path.exists(model_save_base):
        os.makedirs(model_save_base)
    
    early_stopping = EarlyStopping(patience=5, path= os.path.join(model_save_base, 'best_model.pth'))
    
    # iteration
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        print('> Epoch: ', epoch)
        # train epoch
        train_acc, train_prec, train_rec, train_f1, train_loss = train_epoch(train_loader, model, optimizer, 
                                                                                args.arch, args.lb_smooth, device)
        
        # eval 
        test_acc, test_prec, test_rec, test_f1, test_loss = test(test_loader, model, args.arch, device)
        
        pbar.set_description('[Train] Acc: {:.3f}, precision: {:.3f} || [TEST] Acc: {:.3f}, precision {:.3f}'.format(
                                        train_acc, train_prec, test_acc, test_prec))
        
        wandb.log({'loss/train': train_loss, 'loss/test': test_loss, 
                'acc/train': train_acc, 'acc/test': test_acc,
                'prec/train': train_prec, 'prec/test': test_prec,
                'rec/train': train_rec, 'rec/test': test_rec,
                'f1/train': train_f1, 'f1/test': test_f1,
                'lr':optimizer.param_groups[0]['lr']
        })

        if args.lr_decay == 'cosine':
            scheduler.step()
        elif args.lr_decay == 'plateau':
            scheduler.step(test_loss)

        if args.early_stop != 'none':
            if args.early_stop == 'acc':
                early_stopping(-test_acc, model)
            elif args.early_stop == 'prec':
                early_stopping(-test_prec, model)
            elif args.early_stop == 'rec':
                early_stopping(-test_rec, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            model_save_path = os.path.join(model_save_base, 'last_model.pth')
            torch.save(model.state_dict(), model_save_path)

def train_epoch(dataloader, net, optimizer, arch, lb_smooth, device):
    ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=lb_smooth)

    train_losses, train_preds, train_trues = [], [], [] 

    net.train()
    for (img, label) in dataloader:
        img = img.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        if arch == 'pmg':
            input1 = jigsaw_generator(img, 8)
            input2 = jigsaw_generator(img, 4)
            input3 = jigsaw_generator(img, 2)
            
            optimizer.zero_grad()
            out1, _, _, _ = net._forward(input1)
            loss1 = ce_loss(out1, label) * 1
            loss1.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, out2, _, _ = net._forward(input2)
            loss2 = ce_loss(out2, label) * 1
            loss2.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, _, out3, _ = net._forward(input3)
            loss3 = ce_loss(out3, label) * 1
            loss3.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, _, _, out4 = net._forward(img)
            loss4 = ce_loss(out4, label) * 2
            loss4.backward()
            optimizer.step()
            
            _, pred = torch.max(out4, 1)

            loss = loss1 + loss2 + loss3 + loss4 
           
        else:
            out = net(img)
            _, pred = torch.max(out, 1)
            loss = ce_loss(out, label)

            loss.backward()
            optimizer.step()
        

        train_losses.append(loss.item())
        train_trues.extend(label.view(-1).cpu().numpy().tolist())
        train_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())
    
    acc, f1, prec, rec = calculate_metrics(train_trues, train_preds)
    print('trainset result')
    print(confusion_matrix(train_trues, train_preds))

    return acc, prec, rec, f1, np.mean(train_losses)

@torch.no_grad()
def test(dataloader, net, arch, device):
    ce_loss = torch.nn.CrossEntropyLoss()

    test_losses, test_trues, test_preds = [], [], []

    net.eval()
    for (img, label) in dataloader:

        img = img.to(device)
        label = label.to(device)
        
        if arch == 'pmg':
            out1, out2, out3, out_concat = net._forward(img)
            
            loss = ce_loss(out_concat, label)
            _, pred = torch.max(out_concat, 1)

        else:
            out = net(img)
            loss = ce_loss(out, label)
        
            _, pred = torch.max(out, 1) # make prediction

        test_losses.append(loss.item())
        test_trues.extend(label.view(-1).cpu().numpy().tolist())
        test_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())

    print('testset result')
    acc, f1, prec, rec = calculate_metrics(test_trues, test_preds)
    print(confusion_matrix(test_trues, test_preds))

    return acc, prec, rec, f1, np.mean(test_losses)

def load_optimizer(model, arch, lr, lr_decay, weight_decay, epochs, momentum=0.9):
    if arch == 'resnet':
        optimizer = torch.optim.SGD(
                model.parameters(),
                lr = lr,
                momentum=momentum,
                weight_decay = weight_decay,
        )

    elif arch == 'pmg':
        optimizer = torch.optim.SGD([
                {'params': model.classifier_concat.parameters(), 'lr':lr},
                {'params': model.conv_block1.parameters(), 'lr': lr},
                {'params': model.classifier1.parameters(), 'lr': lr},
                {'params': model.conv_block2.parameters(), 'lr': lr},
                {'params': model.classifier2.parameters(), 'lr': lr},
                {'params': model.conv_block3.parameters(), 'lr': lr},
                {'params': model.classifier3.parameters(), 'lr': lr},
                {'params': model.features.parameters(), 'lr': lr/10}
            ],
            momentum=momentum, weight_decay=weight_decay)

    if lr_decay == 'none':
        scheduler = None 
    elif lr_decay == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)
    elif lr_decay == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        NotImplementedError()
    
    return optimizer, scheduler