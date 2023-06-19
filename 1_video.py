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

# Local Imports
from grad_cam_module import GradCAM, GradCamPlusPlus,GradCAM_two_one,GradCAM_two_two
import model.backbone as backbone
import metric.pairsampler as pair
from metric.utils import recall, count_parameters_in_MB, accuracy, AverageMeter
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, SP, SE_Fusion,SoftTarget,Gram_loss
from model.embedding import LinearEmbedding
from TSNdataset import TSNDataSet
from transforms import *

# Get the desired parser
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', type=str, default='UTD', choices=['UTD'])

parser.add_argument('--stu_video_train_path', type=str, default=r"/home/Students/j_n317/SAKDN/data/UTD_rgb_train_list_subject_specific.txt")
parser.add_argument('--stu_video_test_path', type=str, default=r"/home/Students/j_n317/SAKDN/data/UTD_rgb_val_list_subject_specific.txt")
parser.add_argument('--student_base',
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
parser.add_argument('--teacher_base',
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
parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--recall', default=[1], type=int, nargs='+')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')

opts = parser.parse_args()
opts.dataset='UTD_Video_only'

opts.num_classes=27
opts.student_base=backbone.TRN
opts.consensus_type='TRNmultiscale'
# opts.arch='BNInception' #BNInception
opts.arch = 'BNInception'
opts.num_segments=8


opts.epochs=100
opts.lr=0.001
opts.dropout=0.5
opts.lr_decay_epochs=[50]
opts.lr_decay_gamma=0.5
opts.batch=16
opts.img_feature_dim=300

opts.sp_ratio=0    #Similarity preserving distillation   1
opts.st_ratio=0.01       # 0.01
opts.GCAM_ratio=1
opts.semantic_ratio=1

opts.print_freq=27
opts.output_dir='output/'

opts.save_dir= opts.output_dir+'_'.join(map(str, [opts.dataset,'TRN', 
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



# -----------------------------------------------------------------------
#                                Main
# -----------------------------------------------------------------------
def main():

    logging.info('----------- Network Initialization --------------')
    student = opts.student_base(opts.num_classes, opts.num_segments, 'RGB', 
    base_model=opts.arch, consensus_type=opts.consensus_type,  dropout=opts.dropout, img_feature_dim=opts.img_feature_dim, partial_bn=True).cuda()
    teacher = opts.teacher_base(n_classes=opts.num_classes)

    logging.info('Student: %s', student)

    logging.info('Student param size = %fMB', count_parameters_in_MB(student))
    logging.info('-----------------------------------------------')
    UTD_Glove=np.load('/home/Students/j_n317/SAKDN/data/UTD_Glove.npy')
    UTD_Glove=torch.from_numpy(UTD_Glove)
    UTD_Glove=UTD_Glove.float().cuda()

    crop_size = student.crop_size
    scale_size = student.scale_size
    input_mean = student.input_mean
    input_std = student.input_std
    policies = student.get_optim_policies()
    train_augmentation = student.get_augmentation()
    normalize = GroupNormalize(input_mean, input_std)

# ========================= Video Dataset===============================
    video_train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", opts.stu_video_train_path, num_segments=opts.num_segments,
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
        TSNDataSet("", opts.stu_video_test_path, num_segments=opts.num_segments,
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



    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        student = torch.nn.DataParallel(student, device_ids=[0,1]).cuda()


    logging.info("Number of videos in Student Training Set: %d" % len(video_train_loader.dataset))
    logging.info("Number of videos in Student Testing Set: %d" % len(video_test_loader.dataset))

    student = student.cuda()

    optimizer = torch.optim.SGD(policies,
                                opts.lr,
                                momentum=0.9,
                                weight_decay=5e-4)

    cls_criterion=torch.nn.CrossEntropyLoss().cuda()

# =============================Train============================
    def train(s_loader,ep):
        K = opts.recall
        epoch_time = AverageMeter()

        loss_cls= AverageMeter()

        top1_recall = AverageMeter()
        top1_prec = AverageMeter()
        

        student.train()


        cls_loss_all=[]

        loss_all = []
        train_acc=0.
        end = time.time()
        i=1
        torch.cuda.empty_cache() 
        for (s_videos,s_labels) in s_loader:

            s_videos_ablation = torch.zeros(s_videos.size()).cuda()
            s_videos, s_labels = s_videos.cuda(), s_labels.cuda()
            s_videos_combiend=torch.cat((s_videos,s_videos_ablation),0)


            conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,v_semantic,video_output = student(s_videos_combiend)


            s_side=len(s_labels)
            


            video_semantic=UTD_Glove[s_labels]
            video_output=video_output[0:s_side,:]
            v_semantic=v_semantic[0:s_side,:]
            conv_out_conv2=conv_out_conv2[0:s_side,:]
            conv_out_3c=conv_out_3c[0:s_side,:]
            conv_out_4c=conv_out_4c[0:s_side,:]
            conv_out_5a=conv_out_5a[0:s_side,:]
            conv_out_5b=conv_out_5b[0:s_side,:]
            pred=torch.max(video_output,1)[1]
            train_correct=(pred==s_labels).sum()
            train_acc+=train_correct.item()



            cls_loss=cls_criterion(video_output,s_labels)


            loss = cls_loss   

            loss_cls.update(cls_loss.item(), s_videos.size(0))


            rec = recall(video_output, s_labels, K=K)
            prec = accuracy(video_output, s_labels, topk=(1,))
            top1_recall.update(rec[0], s_videos.size(0))
            top1_prec.update(prec[0]/100, s_videos.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cls_loss_all.append(cls_loss.item())

            loss_all.append(loss.item())

            if i % opts.print_freq == 0:
                log_str=('Epoch[{0}]:[{1:03}/{2:03}] '
                        
                        'Cls:{cls_loss.val:.5f}({cls_loss.avg:.4f}) '
                        'recall@1:{top1_recall.val:.2f}({top1_recall.avg:.2f}) '
                        'pre@1:{top1_prec.val:.2f}({top1_prec.avg:.2f}) '.format(
                        ep, i, len(s_loader),cls_loss=loss_cls,top1_recall=top1_recall,top1_prec=top1_prec))
                logging.info(log_str)
            i=i+1
        epoch_time.update(time.time() - end)
        logging.info('[Epoch %d], Loss: %.5f, Training Acc: %.5f\n' %\
            (ep, torch.Tensor(loss_all).mean(), 100*train_acc/(len(s_loader.dataset))))

    
    def eval_student(net, loader, ep):
        torch.cuda.empty_cache() 
        K=opts.recall
        net.eval()
        correct = 0
        embeddings_all= []
        labels_all= []
        with torch.no_grad():
            for i,(images, labels) in enumerate(loader, start=1):
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
            acc = correct/(len(loader.dataset))
            logging.info('[Epoch %d] recall@1: [%.4f]' % (ep, 100 * rec[0]))
            logging.info('[Epoch %d] acc: [%.4f]' % (ep, acc*100))    
        return rec[0], acc, embeddings_all, labels_all


    logging.info('----------- Student Network performance  --------------')

    best_val_recall, best_val_acc, s_prec, labels_all  = eval_student(student, video_test_loader, 0)

    combined_acc=accuracy((s_prec), labels_all, topk=(1,))
    best_combined_acc=combined_acc[0]
    combined_rec = recall((s_prec), labels_all, K=[1])
    best_combined_rec=combined_rec[0]
    logging.info('----------- Combined Network performance  --------------')
    logging.info('Combined acc: [%.4f]\n' % (best_combined_acc)) 
    logging.info('Combined rec: [%.4f]\n' % (best_combined_rec)) 
    for epoch in range(1, opts.epochs+1):
        adjust_learning_rate(optimizer, epoch, opts.lr_decay_epochs)
        train(video_train_loader, epoch)
        val_recall, val_acc, s_prec, labels_all = eval_student(student, video_test_loader, epoch)
        val_combined_acc=accuracy((s_prec), labels_all, topk=(1,))
        val_comb_acc=val_combined_acc[0]
        val_combined_rec = recall((s_prec), labels_all, K=[1])
        val_comb_rec=val_combined_rec[0]
        logging.info('[Epoch %d] combined acc: [%.4f]' % (epoch, val_comb_acc)) 
        logging.info('[Epoch %d] combined rec: [%.4f]\n' % (epoch, val_comb_rec*100)) 
# Save the model       
        if best_val_recall < val_recall:
            best_val_recall = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_recall.pth"))

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_acc.pth"))

        if best_combined_acc < val_comb_acc:
            best_combined_acc = val_comb_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined.pth"))

        if best_combined_rec < val_comb_rec:
            best_combined_rec = val_comb_rec
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined_rec.pth"))        

        F_measure=(2*best_val_acc*best_val_recall)/(best_val_acc+best_val_recall)
        combined_F_measure=(2*best_combined_acc/100*best_combined_rec)/(best_combined_acc/100+best_combined_rec)
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write("Best Test recall@1: %.4f\n" % (best_val_recall * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))
                f.write("Best Test Acc: %.4f\n" % (best_val_acc * 100))
                f.write("Final Acc: %.4f\n" % (val_acc * 100))
                f.write("Best Combined Acc: %.4f\n" % (best_combined_acc))
                f.write("Final Combined Acc: %.4f\n" % (val_comb_acc ))
                f.write("F-measure: %.4f\n" % (F_measure*100))
                f.write("Combined F-measure: %.4f\n" % (combined_F_measure*100))

        logging.info("Best Eval Recall@1: %.4f" % (best_val_recall*100))
        logging.info("Best Eval Acc: %.4f" % (best_val_acc*100))
        logging.info("Best Eval Combined Acc: %.4f" % (best_combined_acc))
        logging.info("Eval F-measure: %.4f" % (F_measure*100))
        logging.info("Eval Combined F-measure: %.4f\n" % (combined_F_measure*100))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = opts.lr * decay
    decay = 5e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()