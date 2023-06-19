# -*- coding: utf-8 -*-
"""

This code reproduces the finding from the above paper
"""
# -------------------------------------------------------------
#                               Imports
# -------------------------------------------------------------

# General Inputs
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
from pickle import TRUE
import sys
import time
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

# Third Party
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# Local Imports
from grad_cam_module import GradCAM, GradCamPlusPlus,GradCAM_two_one,GradCAM_two_two
import model.backbone as backbone
import metric.loss as loss
import metric.pairsampler as pair
from metric.batchsampler import NPairs
from metric.utils import recall, count_parameters_in_MB, accuracy, AverageMeter, DataPrefetcher
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, SP, SE_Fusion,SoftTarget,Gram_loss
from model.embedding import LinearEmbedding
from TSNdataset import TSNDataSet
from transforms import *
from metric.criterion import CRDLoss
from metric.memory import ContrastMemory, AliasMethod
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

# Get the desired parser
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', type=str, default='UTD', choices=['UTD'])
parser.add_argument('--watch_train_path', type=str, default=r"/home/Students/j_n317/DATA/UTD-MHAD/Inertial_a_GASF_subject_specific_train/")
parser.add_argument('--watch_test_path', type=str, default=r"/home/Students/j_n317/DATA/UTD-MHAD/Inertial_a_GASF_subject_specific_test/")
parser.add_argument('--video_train_path', type=str, default=r"/home/Students/j_n317/SAKDN/data/UTD_rgb_train_list_subject_specific.txt")
parser.add_argument('--video_test_path', type=str, default=r"/home/Students/j_n317/SAKDN/data/UTD_rgb_val_list_subject_specific.txt")
parser.add_argument('--modality', type=str, default='a', choices=['a'])
parser.add_argument('--video_base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    SemanticFusionVGG16=backbone.SemanticFusionVGG16,
                                    TSN=backbone.TSN,
                                    TRN=backbone.TRN,
                                    ),
                        default=backbone.ResNet50,
                        action=LookupChoices)
parser.add_argument('--watch_base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    Semantic_ResNet18=backbone.Semantic_ResNet18,
                                    Semantic_ResNet50=backbone.Semantic_ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    ),
                        default=backbone.VggNet16,
                        action=LookupChoices)


parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn','TRN', 'TRNmultiscale'])
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--num_classes', default=27, type=int)
parser.add_argument('--triplet_ratio', default=0, type=float)
parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--angle_ratio', default=0, type=float)
parser.add_argument('--dark_ratio', default=0, type=float)
parser.add_argument('--dark_alpha', default=2, type=float)
parser.add_argument('--dark_beta', default=3, type=float)
parser.add_argument('--at_ratio', default=0, type=float)

parser.add_argument('--triplet_sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.DistanceWeighted,
                    action=LookupChoices)

parser.add_argument('--triplet_margin', type=float, default=0.2)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--iter_per_epoch', default=100, type=int)
parser.add_argument('--lr_decay_epochs', type=int, default=[25, 30, 35], nargs='+')
parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
parser.add_argument('--recall', default=[1], type=int, nargs='+')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')

parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])

parser.add_argument('--trial', type=str, default='1', help='trial id')

parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

opts = parser.parse_args()
opts.dataset='UTD_videotowatch'
opts.modality='a'
opts.num_classes=27
opts.video_base=backbone.TRN
opts.consensus_type='TRNmultiscale'
opts.arch='BNInception' #BNInception
opts.num_segments=8
opts.watch_base=backbone.ResNet18
# opts.watch_base = backbone.VggNet16
opts.sample=pair.DistanceWeighted
opts.loss=loss.L2Triplet

opts.epochs=100
opts.lr=0.0002
opts.dropout=0.5
opts.lr_decay_epochs=[]
opts.lr_decay_gamma=0.5
opts.batch=16

opts.sp_ratio=0    #Similarity preserving distillation   1
opts.st_ratio=0.01       # 0.01
opts.angle_ratio = 1
opts.dist_ratio = 1
opts.GCAM_ratio=1
opts.semantic_ratio=1

opts.print_freq=27
opts.output_dir='output/'
writer = SummaryWriter()


opts.video_load= '/home/Students/j_n317/SAKDN/output/UTD_Video_only_TRN_seg8_epochs100_batch16_lr0.001_dropout0.5/best_acc.pth'
opts.save_dir= opts.output_dir+'_'.join(map(str, [opts.dataset, opts.modality,'DarkLoss_Watch','ResNet18','Video','TRN', 
            'arch',str(opts.arch),'seg'+str(opts.num_segments),'epochs'+str(opts.epochs),'batch'+str(opts.batch), 'lr'+str(opts.lr),'dropout'+str(opts.dropout)]))

if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(opts.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def set_seed(self, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_fn(self):
    np.random.seed(0)

# =========================Watch Dataset===============================
def loadtraindata(data_path):
    path = data_path                                        
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),  
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.CenterCrop(64),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                    #std  = [ 0.229, 0.224, 0.225 ]),
                                                ])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch,
                                              shuffle=True, num_workers=3, worker_init_fn=_init_fn)
    return trainloader

def loadtestdata(data_path):
    path = data_path
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)), 
                                                    #transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                    #std  = [ 0.229, 0.224, 0.225 ]),
                                                    ])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=opts.batch,
                                             shuffle=False, num_workers=3, worker_init_fn=_init_fn)
    return testloader


# -----------------------------------------------------------------------
#                                Main
# -----------------------------------------------------------------------
def main():

    logging.info('----------- Network Initialization --------------')
    video = opts.video_base(opts.num_classes, opts.num_segments, 'RGB', base_model=opts.arch, 
                            consensus_type=opts.consensus_type,  dropout=opts.dropout, partial_bn=True).cuda()
    watch = opts.watch_base(n_classes=opts.num_classes).cuda()
    
    logging.info('Watch: %s', watch)
    logging.info('Video: %s', video)
    logging.info('Watch param size = %fMB', count_parameters_in_MB(watch))
    logging.info('Video param size = %fMB', count_parameters_in_MB(video))
    logging.info('-----------------------------------------------')
    watch_train_loader = loadtraindata(opts.watch_train_path)
    watch_test_loader = loadtestdata(opts.watch_test_path)
    UTD_Glove=np.load('/home/Students/j_n317/SAKDN/data/UTD_Glove.npy')
    UTD_Glove=torch.from_numpy(UTD_Glove)
    UTD_Glove=UTD_Glove.float().cuda()
    torch.cuda.empty_cache()

# ========================= Video Dataset===============================
    crop_size = video.crop_size
    scale_size = video.scale_size
    input_mean = video.input_mean
    input_std = video.input_std
    train_augmentation = video.get_augmentation()
    normalize = GroupNormalize(input_mean, input_std)

    video_train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", opts.video_train_path, num_segments=opts.num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl="{:06d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=opts.arch == 'BNInception'),
                       ToTorchFormatTensor(div=opts.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=opts.batch, shuffle=True, num_workers=3, pin_memory=False,worker_init_fn=_init_fn)

    video_test_loader = torch.utils.data.DataLoader(
        TSNDataSet("", opts.video_test_path, num_segments=opts.num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl="{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=opts.arch == 'BNInception'),
                       ToTorchFormatTensor(div=opts.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=opts.batch, shuffle=False, num_workers=3, pin_memory=False,worker_init_fn=_init_fn)  


    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        print("*"*100)
        print("THE VISIBLE DEVICES ARE",os.environ['CUDA_VISIBLE_DEVICES'])
        print("*"*100)

        devices = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'])-1))
        
        video = torch.nn.DataParallel(video, device_ids=devices).cuda()
        watch = torch.nn.DataParallel(watch, device_ids=devices).cuda()
    else:
        video = video.cuda()
        watch = watch.cuda()

    logging.info("Number of images in Teacher Training Set: %d" % len(watch_train_loader.dataset))
    logging.info("Number of images in Teacher Testing Set: %d" % len(watch_test_loader.dataset))
    logging.info("Number of videos in Student Training Set: %d" % len(video_train_loader.dataset))
    logging.info("Number of videos in Student Testing Set: %d" % len(video_test_loader.dataset))

    if opts.load is not None:
        video.load_state_dict(torch.load(opts.load))
        logging.info("Loaded Model from %s" % opts.load)
    
    video.load_state_dict(torch.load(opts.video_load))
    video = video.cuda()
    watch = watch.cuda()

    optimizer = optim.Adam(watch.parameters(), lr=opts.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)
# Loss function   
    st_criterion = SoftTarget(4).cuda()
    cls_criterion=torch.nn.CrossEntropyLoss().cuda()
    dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)
    dist_criterion = RkdDistance().cuda()
    angle_criterion = RKdAngle().cuda()
    
    at_criterion = AttentionTransfer()
    criterion_semantic=torch.nn.MSELoss().cuda()
    sp_criterion=SP().cuda()  # Similarity perserving
# =============================Train============================

    def train (video_loader, watch_loader, ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        epoch_time = AverageMeter()
        loss_st= AverageMeter()
        loss_cls= AverageMeter()
        loss_semantic= AverageMeter()
        loss_dist = AverageMeter()
        loss_at = AverageMeter()
        loss_sp = AverageMeter()
        loss_angle= AverageMeter()
        top1_recall = AverageMeter()
        top1_prec = AverageMeter()       
        video.eval()
        watch.train()
        
        dist_loss_all = []
        angle_loss_all = []
        at_loss_all = []
        cls_loss_all=[]
        st_loss_all = []
        sp_loss_all = []
        dark_loss_all = []
        semantic_loss_all = []
        loss_all = []
        train_acc=0.
        end = time.time()
        i=1
        torch.cuda.empty_cache() 

        for (t_videos,t_labels), (s_images, s_labels) in zip(video_loader, watch_loader):
            data_time.update(time.time() - end)
            s_images_ablation = torch.zeros(s_images.size()).cuda()
            t_videos_ablation = torch.zeros(t_videos.size()).cuda()
            s_images, s_labels = s_images.cuda(), s_labels.cuda()
            t_videos, t_labels = t_videos.cuda(), t_labels.cuda()
            s_images_combined=torch.cat((s_images,s_images_ablation),0)
            t_videos_combined=torch.cat((t_videos,t_videos_ablation),0)


            with torch.no_grad():
                conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,v_semantic,video_output = video(t_videos_combined)

            t_out3,t_out4,t_out5,t_out6,t_out7_,t_out7, t_out8 = watch(s_images_combined, True)

            watch_side=len(s_labels)
            video_side=len(t_labels)

            soft_video_output = F.softmax(video_output, dim=1)
            soft_watch_out8 = F.softmax(t_out8, dim=1)
            soft_video_predict=torch.max(soft_video_output,1)[1]
            soft_watch_predict=torch.max(soft_watch_out8,1)[1]
            video_semantic=UTD_Glove[soft_video_predict]
            watch_semantic=UTD_Glove[soft_watch_predict]

            video_output=video_output[0:video_side,:]
            v_semantic=v_semantic[0:video_side,:]
            conv_out_conv2=conv_out_conv2[0:video_side,:]

            t_out3=t_out3[0:watch_side,:]
            t_out4=t_out4[0:watch_side,:]
            t_out5=t_out5[0:watch_side,:]
            t_out7=t_out7[0:watch_side,:]
            t_out8=t_out8[0:watch_side,:]


            pred_video=torch.max(video_output,1)[1]
            pred = torch.max(t_out8,1)[1]

            train_correct=(pred==s_labels).sum()
            train_acc+=train_correct.item()

            # dist_loss = opts.dist_ratio * (dist_criterion(v_semantic, t_out7))
            # angle_loss = opts.angle_ratio * (angle_criterion(v_semantic, t_out7))

            sp_loss= opts.sp_ratio * (sp_criterion( t_out8,video_output))
            st_loss= opts.st_ratio * (st_criterion(t_out8,video_output)) # Soft Target 
            # semantic_loss=opts.semantic_ratio*(criterion_semantic(t_out7, v_semantic)) # Semantic loss
            at_loss = at_criterion (t_out7, v_semantic)
            cls_loss=cls_criterion(t_out8,s_labels)
            loss = cls_loss + st_loss + at_loss
            # + dist_loss + angle_loss + semantic_loss
            #  + dist_loss
            #  
            # cls: entropy; st_loss: soft target; sp: similarity perserving

            # loss_dist.update(dist_loss.item(), t_videos.size(0))
            # loss_angle.update(angle_loss.item(), t_videos.size(0))
            loss_at.update(at_loss.item(), t_videos.size(0))
            loss_sp.update(sp_loss.item(), t_videos.size(0))
            loss_st.update(st_loss.item(), t_videos.size(0))
            loss_cls.update(cls_loss.item(), t_videos.size(0))
            # loss_semantic.update(semantic_loss.item(), t_videos.size(0))

            rec = recall(t_out8, s_labels, K=K)
            prec = accuracy(t_out8, s_labels, topk=(1,))
            top1_recall.update(rec[0], s_images.size(0))
            top1_prec.update(prec[0]/100, s_images.size(0))
            top1_recall.update(rec[0], s_images.size(0))
            top1_prec.update(prec[0]/100, s_images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            cls_loss_all.append(cls_loss.item())
            st_loss_all.append(st_loss.item())
            at_loss_all.append(at_loss.item())
            # dist_loss_all.append(dist_loss.item())
            # angle_loss_all.append(angle_loss.item())
            # semantic_loss_all.append(semantic_loss.item())
            sp_loss_all.append(sp_loss.item())
            loss_all.append(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opts.print_freq == 0:
                log_str=('Epoch[{0}]:[{1:03}/{2:03}] '                       
                        'St:{st_loss.val:.5f} ({st_loss.avg:.4f}) '
                        'Cls:{cls_loss.val:.5f}({cls_loss.avg:.4f}) '
                        'recall@1:{top1_recall.val:.2f}({top1_recall.avg:.2f}) '
                        'pre@1:{top1_prec.val:.2f}({top1_prec.avg:.2f}) '.format(
                        ep, i, len(video_loader),st_loss=loss_st,cls_loss=loss_cls,top1_recall=top1_recall,top1_prec=top1_prec))
                logging.info(log_str)

            i=i+1
        epoch_time.update(time.time() - end)
        logging.info('[Epoch %d] Time:  %.5f, Loss: %.5f, St:  %.5f, Acc: %.5f\n' %\
            (ep, epoch_time.val, torch.Tensor(loss_all).mean(), torch.Tensor(st_loss_all).mean(),100*train_acc/(len(video_loader.dataset))))

# =============================Eval Video and Watch============================
    
    def eval_video(net, video_loader, ep):
        torch.cuda.empty_cache() 
        K=opts.recall
        net.eval()
        correct = 0
        embeddings_all= []
        labels_all= []
        with torch.no_grad():
            for i,(images, labels) in enumerate(video_loader, start=1):
                images, labels = images.cuda(), labels.cuda()
                conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,\
                conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,video_semantic,video_output = net(images)
                pred=torch.max(video_output,1)[1]
                num_correct=(pred==labels).sum()
                correct+=num_correct.item()
                embeddings_all.append(video_output.data)
                labels_all.append(labels.data)  
            embeddings_all = torch.cat(embeddings_all).cpu()
            labels_all = torch.cat(labels_all).cpu()
            rec = recall(embeddings_all, labels_all, K=K)
            acc = correct/(len(video_loader.dataset))
            logging.info('[Epoch %d] recall@1: [%.4f]' % (ep, 100 * rec[0]))
            logging.info('[Epoch %d] acc: [%.4f]' % (ep, acc*100))    
        return embeddings_all

    def eval_watch(net, watch_loader, ep):
        
        torch.cuda.empty_cache() 
        K = opts.recall
        net.eval()
        t_embeddings_all,t_labels_all = [], []
        rec = []
        t_correct = 0

        with torch.no_grad():
            for (s_images, s_labels) in watch_loader:
                s_images, s_labels = s_images.cuda(), s_labels.cuda()

                t_embedding = net(s_images)
                t_pred=torch.max(t_embedding,1)[1]
                t_num_correct=(t_pred==s_labels).sum()
                t_correct+=t_num_correct.item()
                t_embeddings_all.append(t_embedding.data)
                t_labels_all.append(s_labels.data)

            t_embeddings_all = torch.cat(t_embeddings_all).cpu()
            t_labels_all = torch.cat(t_labels_all).cpu()
            rec = recall(t_embeddings_all, t_labels_all, K=K)
            prec = accuracy(t_embeddings_all, t_labels_all, topk=(1,))
            t_acc = t_correct/(len(watch_loader.dataset))

            logging.info('[Epoch %d] Watch acc: [%.4f]  ' % (ep, t_acc*100))    
        return rec[0], t_acc, t_embeddings_all, t_labels_all

    logging.info('----------- Video Network performance  --------------')
    video_prec = eval_video(video, video_test_loader, 0)

    logging.info('----------- Watch Network performance --------------')
    best_val_recall, watch_eval_acc, watch_prec, watch_labels_all =eval_watch(watch, watch_test_loader, 0)


    combined_acc=accuracy((watch_prec+video_prec), watch_labels_all, topk=(1,))
    best_combined_acc=combined_acc[0]
    combined_rec = recall((watch_prec+video_prec), watch_labels_all, K=[1])
    best_combined_rec=combined_rec[0]

    logging.info('----------- Combined Network performance  --------------')
    logging.info('Combined acc: [%.4f]\n' % (best_combined_acc)) 
    logging.info('Combined rec: [%.4f]\n' % (best_combined_rec)) 

    for epoch in range(1, opts.epochs+1):
        train(video_train_loader, watch_train_loader,epoch)
        val_recall,val_acc, watch_prec, labels_all = eval_watch (watch, watch_test_loader, epoch)
        
        val_combined_acc=accuracy((watch_prec+video_prec), labels_all, topk=(1,))
        val_comb_acc=val_combined_acc[0]
        val_combined_rec = recall((watch_prec+video_prec), labels_all, K=[1])
        # val_comb_rec=val_combined_rec[0]
        logging.info('[Epoch %d] combined acc: [%.4f]' % (epoch, val_comb_acc)) 
        # logging.info('[Epoch %d] combined rec: [%.4f]\n' % (epoch, val_comb_rec*100)) 
# Save the model
        if best_val_recall < val_recall:
            best_val_recall = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(video.state_dict(), "%s/%s"%(opts.save_dir, "best_recall.pth"))

        if watch_eval_acc < val_acc:
            watch_eval_acc = val_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(watch.state_dict(), "%s/%s"%(opts.save_dir, "best_acc.pth"))

        if best_combined_acc < val_comb_acc:
            best_combined_acc = val_comb_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(watch.state_dict(), "%s/%s"%(opts.save_dir, "best_combined.pth"))

        # if best_combined_rec < val_comb_rec:
        #     best_combined_rec = val_comb_rec
        #     if opts.save_dir is not None:
        #         if not os.path.isdir(opts.save_dir):
        #             os.mkdir(opts.save_dir)
        #         torch.save(video.state_dict(), "%s/%s"%(opts.save_dir, "best_combined_rec.pth"))        

        F_measure=(2*watch_eval_acc*best_val_recall)/(watch_eval_acc+best_val_recall)
        # combined_F_measure=(2*best_combined_acc/100*best_combined_rec)/(best_combined_acc/100+best_combined_rec)
        
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(video.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write("Best Test recall@1: %.4f\n" % (best_val_recall * 100))
                # f.write("Final Recall@1: %.4f\n" % (val_recall * 100))
                f.write("Best Test Acc: %.4f\n" % (watch_eval_acc * 100))
                f.write("Final Acc: %.4f\n" % (val_acc * 100))
                f.write("Best Combined Acc: %.4f\n" % (best_combined_acc))
                f.write("Final Combined Acc: %.4f\n" % (val_comb_acc ))
                f.write("F-measure: %.4f\n" % (F_measure*100))
                # f.write("Combined F-measure: %.4f\n" % (combined_F_measure*100))

        logging.info("Best Eval Recall@1: %.4f" % (best_val_recall*100))
        logging.info("Best Eval Acc: %.4f" % (watch_eval_acc*100))
        logging.info("Best Eval Combined Acc: %.4f" % (best_combined_acc))
        logging.info("Eval F-measure: %.4f" % (F_measure*100))
        # logging.info("Eval Combined F-measure: %.4f\n" % (combined_F_measure*100))


if __name__ == '__main__':
    main()
