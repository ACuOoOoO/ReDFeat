import argparse

import numpy as np

import os

import shutil

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from torchvision import transforms
# from lib.dataset import MegaDepthDataset
from lib.dataset import MMDataset
# from lib.loss_full import MMLoss
# from lib.model_full import MMNet
from lib.loss import MMLoss
from lib.model import MMNet
import warnings
import os


warnings.filterwarnings("ignore", category=UserWarning) 

# Define epoch function
def process_epoch(
        epoch_idx,
        model, loss_function, optimizer, dataloader,
        log_file, args, train=True, scheduler=None,
):  
    running_loss_desc = 0
    running_loss_peak = 0
    running_loss_rep = 0
    epoch_losses = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for batch_idx, batch in progress_bar:
        flag = True
        model.train()
        batch['train'] = train
        batch['epoch_idx'] = epoch_idx
        batch['batch_idx'] = batch_idx
        batch['batch_size'] = args.batch_size
        batch['log_interval'] = args.log_interval
        img1 = batch.pop('img1').cuda(non_blocking=True)
        img2 = batch.pop('img2').cuda(non_blocking=True)
        valid_im = torch.ones(img1.shape[0],device=img1.device).bool()
        aflow = batch.pop('aflow').cuda(non_blocking=True)
        for i in range(aflow.shape[0]):
            aflow_i = aflow[i]
            valid_points = (aflow_i<1000)*1.0/2
            if valid_points.sum()<192*192*1.0/4:
                print('hello')
                valid_im[i]=False
                aflow.pop(i)
        img1 = img1[valid_im]
        img2 = img2[valid_im]
        if len(aflow)!=0:
            output = model(img1,img2)
            feat1 = output['feat'][0]
            feat2 = output['feat'][1]
            score1 = output['score'][0]
            score2 = output['score'][1]

            # img2_warp = KGT.remap(img2,aflow[...,0],aflow[...,1])
            # mask = img2_warp<0.001
            # t = img1*mask+img2_warp
            # t = TF.to_pil_image(t[0])
            # t.save('t.png')
            loss = loss_function(feat1,score1,feat2,score2,aflow,img1,img2)
            optimizer.zero_grad()
            (loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            current_loss = loss.item()
            epoch_losses.append(current_loss)
            

        running_loss_desc += loss_function.loss_desc_
        running_loss_peak += loss_function.loss_peak_
        running_loss_rep += loss_function.loss_rep_
        # running_pos += loss_function.pos
        # running_neg += loss_function.neg
        # running_score += loss_function.score_mean


        progress_bar.set_postfix(epoch = epoch_idx, loss=('%.4f' % np.mean(epoch_losses)))

        if batch_idx % args.log_interval == 0:
            log_file.write('[%s] epoch %d - batch %d / %d - avg_loss: %f\n' % (
                'train' if train else 'valid',
                epoch_idx, batch_idx, len(dataloader), np.mean(epoch_losses)
            ))
    print('loss_desc: {}\n'.format(running_loss_desc.item()/(batch_idx+1)))
    print('loss_peak: {}\n'.format(running_loss_peak.item()/(batch_idx+1)))
    print('loss_rep: {}\n'.format(running_loss_rep.item()/(batch_idx+1)))
    # print('pos: {}\n'.format(running_pos.item()/(batch_idx+1)))
    # print('neg: {}\n'.format(running_neg.item()/(batch_idx+1)))
    # print('score: {}\n'.format(running_score.item()/(batch_idx+1)))
    log_file.write('[%s] epoch %d - avg_loss: %f   loss_desc: %f   loss_rep: %f   loss_peak: %f \n ' % (
        'train' if train else 'valid',
        epoch_idx,
        np.mean(epoch_losses),
        running_loss_desc.item()/(batch_idx+1),
        running_loss_rep.item()/(batch_idx+1),
        running_loss_peak.item()/(batch_idx+1),
        
    ))
    log_file.flush()
    scheduler.step()
    return np.mean(epoch_losses)

def train(args):
    # Creating CNN model
    model = MMNet()
    model = model.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_epochs, eta_min=1e-7)
    # Create loss fcuntion
    loss = MMLoss(lam1=args.lam1,lam2=args.lam2)

    
    # Create dataloader
    if args.image_type == 'VIS_SAR':
        training_dataset = MMDataset(args.datapath,args.image_type,transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.RandomAdjustSharpness(2, p=0.5),
                                                                        transforms.Resize(256)
                                                                         ]))
    elif args.image_type == 'VIS_IR':
        training_dataset = MMDataset(args.datapath,args.image_type,transform=transforms.Compose([transforms.ToTensor(),
                                                                        #transforms.RandomAdjustSharpness(2, p=0.5)
                                                                         ]))
    elif args.image_type == 'VIS_NIR':
        training_dataset = MMDataset(args.datapath,args.image_type,transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.RandomAdjustSharpness(2, p=0.5),
                                                                        transforms.Resize(256)
                                                                        ]))

    else:
        print('[Error] Invalid Image Type.')

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        prefetch_factor=4,
    )
    
    # Create the checkpoint 7directory
    if os.path.isdir(args.name):
        print('[Warning] Checkpoint directory already exists.')
    else:
        os.mkdir(args.name)
        
    # Open the log file for writing
    if os.path.exists(args.log_file):
        print('[Warning] Log file already exists.')
    log_file = open(args.log_file, 'a+')

    # Initialize the history
    train_loss_history = []

    # Start the training
    for epoch_idx in range(1, args.num_epochs + 1):
        train_loss_history.append(
            process_epoch(
                epoch_idx,
                model, loss, optimizer, training_dataloader,
                log_file, args, scheduler=scheduler
            )
        )
        
        # Save the current checkpoint
        checkpoint_path = os.path.join(
            args.name,
            '%02d.pth' % (epoch_idx)
        )
        checkpoint = {
            'args': args,
            'epoch_idx': epoch_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if epoch_idx%10==0 and args.image_type!='SAR':
            torch.save(checkpoint, checkpoint_path)
        elif args.image_type=='SAR' or args.image_type=='vis':
            torch.save(checkpoint, checkpoint_path)
        
        
    # Close the log file 
    log_file.close()


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument(
        '--num_epochs', type=int, default=50,
        help='number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='initial learning rate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='batch size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='number of workers for data loading'
    )

    parser.add_argument(
        '--log_interval', type=int, default=250,
        help='loss logging interval'
    )

    parser.add_argument(
        '--log_file', type=str, default='log.txt',
        help='loss logging file'
    )

    parser.add_argument(
        '--name', type=str, default='ReDFeat_SAR',
        help='directory for training checkpoints'
    )
    

    parser.add_argument(
        '--image_type', type=str, default='VIS_NIR',
        help='type of training images VIS_IR, VIS_NIR, VIS_SAR'
    )

    parser.add_argument(
        '--datapath', type=str, default='Multimodal_Feature_Evaluation-main',
        help='root for training data'
    )

    parser.add_argument(
        '--gpu', type=int, default=0,
        help='gpu id'
    )

    parser.add_argument(
        '--lam1', type=float, default=1.0,
        help='weight for peaking loss'
    )
    parser.add_argument(
        '--lam2', type=float, default=8.0,
        help='weight for repeatability loss'
    )
    args = parser.parse_args()
    print(args)

    # CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    # Seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    train(args)

