import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import BCP_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d
from utils.BCP_utils import update_ema_students 

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/autodl-fs/DARC-main/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DARC_fixed', help='experiment_name') 
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default=6.0, help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')

# <--- RiCo 权重设置 ---
parser.add_argument('--rico_weight', type=float,
                    default=200.0, help='MAX weight for rico loss (will be ramped up)')
# <--- PACL (Performance-Adaptive Curriculum Learning) 参数 --->
parser.add_argument('--warmup_ratio', type=float, default=0.3, help='热身期占总 epoch 的比例 (Tw)')
parser.add_argument('--sprint_ratio', type=float, default=0.7, help='冲刺期开始占总 epoch 的比例 (Ts)')
parser.add_argument('--beta', type=float, default=0.6, help='适应期难度增长的缩放因子 (Beta)')
# --- DARC 补充参数：对抗扰动、EMA 和稳定 winner 判定 ---
parser.add_argument('--adv_weight', type=float, default=0.05, help='MAX weight for uncertainty-guided adversarial loss')
parser.add_argument('--adv_eps_min', type=float, default=0.001, help='minimum FGSM epsilon for low-uncertainty regions')
parser.add_argument('--adv_eps_max', type=float, default=0.03, help='maximum FGSM epsilon for high-uncertainty regions')
parser.add_argument('--adv_rampup', type=float, default=5000.0, help='ramp-up iterations for adversarial loss')
parser.add_argument('--rico_rampup', type=float, default=5000.0, help='ramp-up iterations for RiCo loss')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for the teacher model')
parser.add_argument('--winner_momentum', type=float, default=0.9, help='moving-average momentum for winner determination')


args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c * torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

# <--- NEW: 真实的 PACL 性能感知自适应课程学习 --->
def get_adaptive_bcp_ratio(step, max_steps, p_val, args):
    """
    PACL 三阶段难度控制。
    注意：原脚本按 epoch_num 调度，实际论文写的是按训练进度/iteration。
    这里改为按 iter_num/max_iterations 调度，使 Tw/Ts 与论文中的训练阶段一致。
    p_val: EMA teacher 最近一次验证 Dice，范围通常为 [0, 1]。
    """
    T_w = max_steps * args.warmup_ratio
    T_s = max_steps * args.sprint_ratio
    beta = args.beta
    p_val = float(np.clip(p_val, 0.0, 1.0))

    if step <= T_w:
        gamma = (step / max(T_w, 1.0)) * 0.2
    elif step <= T_s:
        progress = (step - T_w) / max(T_s - T_w, 1.0)
        gamma = 0.2 + progress * beta * (0.5 + 0.5 * p_val)
    else:
        # Sprint 阶段继续小幅推高难度，而不是在 Ts 后完全固定。
        progress = (step - T_s) / max(max_steps - T_s, 1.0)
        gamma_ts = 0.2 + beta * (0.5 + 0.5 * p_val)
        gamma = gamma_ts + (1.0 - gamma_ts) * progress

    gamma = min(1.0, max(0.0, gamma))

    R_sq = 0.3 + 0.3 * gamma
    R_st = 0.2 + 0.3 * gamma
    return R_sq, R_st

# <--- MODIFIED: 支持动态 ratio，默认为 "简单" 模式 (用于 pre_train) ---
def generate_mask(img, ratio=0.3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    
    # 动态计算 patch 大小
    patch_x = int(img_x * ratio)
    patch_y = int(img_y * ratio)
    
    # 边界保护
    patch_x = min(max(patch_x, 16), img_x - 1)
    patch_y = min(max(patch_y, 16), img_y - 1)

    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

# <--- MODIFIED: 支持动态 ratio，默认为 "简单" 模式 (用于 pre_train) ---
def contact_mask(img, ratio=0.2, orientation='random'):
    """生成条带掩码。
    修正点：原实现用 h 从 y 轴采样，却写成 mask[h:h+patch_y, :]，
    当 H/W 不相等时会遮错方向；即使 256x256，也会使“条带宽度”语义混乱。
    orientation='vertical'  遮挡若干列；orientation='horizontal' 遮挡若干行。
    """
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y, device=img.device)
    mask = torch.ones(img_x, img_y, device=img.device)

    if orientation == 'random':
        orientation = 'vertical' if random.random() < 0.5 else 'horizontal'

    if orientation == 'vertical':
        patch_w = min(max(int(img_y * ratio), 16), img_y - 1)
        start = np.random.randint(0, img_y - patch_w)
        mask[:, start:start + patch_w] = 0
        loss_mask[:, :, start:start + patch_w] = 0
    else:
        patch_h = min(max(int(img_x * ratio), 16), img_x - 1)
        start = np.random.randint(0, img_x - patch_h)
        mask[start:start + patch_h, :] = 0
        loss_mask[:, start:start + patch_h, :] = 0

    return mask.long(), loss_mask.long()


def update_ema_two_students(model_s1, model_s2, ema_model, alpha=0.99):
    """Teacher = 两个学生参数均值的 EMA。
    比外部黑盒 update_ema_students 更利于论文复现，并且跳过 long/int buffer。
    """
    with torch.no_grad():
        s1_state = model_s1.state_dict()
        s2_state = model_s2.state_dict()
        ema_state = ema_model.state_dict()
        for key in ema_state.keys():
            if torch.is_floating_point(ema_state[key]):
                student_avg = 0.5 * (s1_state[key].detach() + s2_state[key].detach())
                ema_state[key].mul_(alpha).add_(student_avg, alpha=1.0 - alpha)
            else:
                # 例如 BatchNorm 的 num_batches_tracked，直接保留其中一个学生的 buffer。
                ema_state[key].copy_(s1_state[key])


def disagreement_uncertainty(logits_s1, logits_s2, eps=1e-6):
    """基于双学生预测差异的不确定性图，输出 shape: [B,1,H,W]，范围 [0,1]。"""
    p1 = F.softmax(logits_s1.detach(), dim=1)
    p2 = F.softmax(logits_s2.detach(), dim=1)
    u = torch.mean(torch.abs(p1 - p2), dim=1, keepdim=True)
    b = u.shape[0]
    u_flat = u.view(b, -1)
    u_min = u_flat.min(dim=1)[0].view(b, 1, 1, 1)
    u_max = u_flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (u - u_min) / (u_max - u_min + eps)


def make_uncertainty_fgsm(model, x, pseudo_label, uncertainty, eps_min=0.001, eps_max=0.03):
    """不确定性驱动 FGSM。
    高不确定区域使用更大 epsilon；低不确定区域使用更小 epsilon。
    """
    model_was_training = model.training
    model.eval()

    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, pseudo_label.long())
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

    if uncertainty.shape[-2:] != x.shape[-2:]:
        uncertainty = F.interpolate(uncertainty, size=x.shape[-2:], mode='bilinear', align_corners=False)
    eps_map = eps_min + (eps_max - eps_min) * uncertainty.detach()
    x_adv = x_adv + eps_map * grad.sign()

    # ACDC 预处理通常已归一化，这里用当前 batch 的动态范围裁剪，避免破坏强度尺度。
    x_min = x.detach().amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.detach().amax(dim=(1, 2, 3), keepdim=True)
    x_adv = torch.max(torch.min(x_adv, x_max), x_min).detach()

    if model_was_training:
        model.train()
    return x_adv


def hard_dice_score(logits, label, num_classes=4, eps=1e-6):
    """用于 FA-RC winner 判定的前景 Dice，排除背景类。"""
    pred = torch.argmax(F.softmax(logits.detach(), dim=1), dim=1)
    scores = []
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        label_c = (label == c).float()
        inter = (pred_c * label_c).sum()
        denom = pred_c.sum() + label_c.sum()
        if denom > 0:
            scores.append((2.0 * inter + eps) / (denom + eps))
    if len(scores) == 0:
        return 0.0
    return torch.stack(scores).mean().item()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            
            # 预训练使用默认的简单难度 (generate_mask 默认 ratio=0.3)
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    # <--- RiCo ---
    model = BCP_net(in_chns=1, class_num=num_classes)
    model_rival = BCP_net(in_chns=1, class_num=num_classes) # <--- RiCo ---
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # <--- RiCo ---
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_rival = optim.SGD(model_rival.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) # <--- RiCo ---
    
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    load_net_opt(model_rival, optimizer_rival, pre_trained_model) # <--- RiCo ---
    
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    # <--- 可视化保存路径 ---
    vis_save_path = os.path.join(snapshot_path, "visualizations")
    os.makedirs(vis_save_path, exist_ok=True)
    logging.info(f"Visualizations will be saved to: {vis_save_path}")

    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    model_rival.train() # <--- RiCo ---
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance2 = 0.0 
    best_performance_ema = 0.0
    best_hd = 100

    # <--- PACL: 初始化最新验证集性能 --->
    latest_p_val = 0.0
    
    # <--- 可视化追踪变量 ---
    loss_s1_history = []
    loss_s2_history = []
    s1_wins = 0
    s2_wins = 0
    winner_dice_ema_s1 = None
    winner_dice_ema_s2 = None
    
    vis_save_path = os.path.join(snapshot_path, "visualizations")
    os.makedirs(vis_save_path, exist_ok=True)
    logging.info(f"Visualizations will be saved to: {vis_save_path}")

    # <--- NEW: 初始化用于保存 loss 和 dice 的 txt 文件 ---
    metrics_record_file = os.path.join(snapshot_path, "metrics_record.txt")
    with open(metrics_record_file, "w") as f:
        # 写入表头
        f.write("Iter,Total_Loss,Loss_S1,Loss_S2,Loss_RiCo,Dice_S1,Dice_S2,Dice_EMA\n")

    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                
                # <--- MODIFIED: 自适应混合难度 (Curriculum Learning) ---
                # 1. 获取当前 epoch 的动态难度参数
                ratio_sq, ratio_st = get_adaptive_bcp_ratio(iter_num, max_iterations, latest_p_val, args)
                
                # 2. 将动态参数传入 Mask 生成函数
                if random.random() < 0.5:
                    # S1 执行 "方形" (Square), S2 执行 "条形" (Strip)
                    img_mask_s1, loss_mask_s1 = generate_mask(img_a, ratio=ratio_sq)
                    img_mask_s2, loss_mask_s2 = contact_mask(img_a, ratio=ratio_st)
                else:
                    # 互换任务
                    img_mask_s1, loss_mask_s1 = contact_mask(img_a, ratio=ratio_st)
                    img_mask_s2, loss_mask_s2 = generate_mask(img_a, ratio=ratio_sq)
                # -------------------------------------------------

                unl_label = ulab_a * img_mask_s1 + lab_a * (1 - img_mask_s1)
                l_label = lab_b * img_mask_s1 + ulab_b * (1 - img_mask_s1)
            
            consistency_weight = get_current_consistency_weight(epoch_num)
            
            # <--- RiCo Ramp-Up ---
            rampup_factor = ramps.sigmoid_rampup(iter_num, args.rico_rampup)
            rico_weight_ramped = args.rico_weight * rampup_factor
            adv_weight_ramped = args.adv_weight * ramps.sigmoid_rampup(iter_num, args.adv_rampup)

            # <--- S1 混合输入 ---
            net_input_unl_s1 = uimg_a * img_mask_s1 + img_a * (1 - img_mask_s1)
            net_input_l_s1 = img_b * img_mask_s1 + uimg_b * (1 - img_mask_s1)

            # <--- S2 混合输入 ---
            net_input_unl_s2 = uimg_a * img_mask_s2 + img_a * (1 - img_mask_s2)
            net_input_l_s2 = img_b * img_mask_s2 + uimg_b * (1 - img_mask_s2)

            # --- 学生 1 (model) ---
            out_unl_s1 = model(net_input_unl_s1)
            out_l_s1 = model(net_input_l_s1)
            unl_dice_s1, unl_ce_s1 = mix_loss(out_unl_s1, plab_a, lab_a, loss_mask_s1, u_weight=args.u_weight, unlab=True)
            l_dice_s1, l_ce_s1 = mix_loss(out_l_s1, lab_b, plab_b, loss_mask_s1, u_weight=args.u_weight)

            loss_ce_s1 = unl_ce_s1 + l_ce_s1 
            loss_dice_s1 = unl_dice_s1 + l_dice_s1
            loss_s1 = (loss_dice_s1 + loss_ce_s1) / 2            

            # --- 学生 2 (model_rival) ---
            out_unl_s2 = model_rival(net_input_unl_s2)
            out_l_s2 = model_rival(net_input_l_s2)
            unl_dice_s2, unl_ce_s2 = mix_loss(out_unl_s2, plab_a, lab_a, loss_mask_s2, u_weight=args.u_weight, unlab=True)
            l_dice_s2, l_ce_s2 = mix_loss(out_l_s2, lab_b, plab_b, loss_mask_s2, u_weight=args.u_weight)

            loss_ce_s2 = unl_ce_s2 + l_ce_s2 
            loss_dice_s2 = unl_dice_s2 + l_dice_s2
            loss_s2 = (loss_dice_s2 + loss_ce_s2) / 2

            # --- RiCo 竞争 ---
            out_u_s1 = model(uimg_a)
            out_u_s2 = model_rival(uimg_a)
            out_u_soft_s1 = F.softmax(out_u_s1, dim=1)
            out_u_soft_s2 = F.softmax(out_u_s2, dim=1)

            # --- TD-AP: 不确定性驱动对抗扰动 ---
            # 原脚本只有双学生分歧和 RiCo，没有真正把 uncertainty adversarial attack 加入训练。
            with torch.no_grad():
                uncertainty_u = disagreement_uncertainty(out_u_s1, out_u_s2)

            x_adv_s1 = make_uncertainty_fgsm(
                model, uimg_a, plab_a, uncertainty_u,
                eps_min=args.adv_eps_min, eps_max=args.adv_eps_max
            )
            x_adv_s2 = make_uncertainty_fgsm(
                model_rival, uimg_a, plab_a, uncertainty_u,
                eps_min=args.adv_eps_min, eps_max=args.adv_eps_max
            )
            adv_logits_s1 = model(x_adv_s1)
            adv_logits_s2 = model_rival(x_adv_s2)
            loss_adv_s1 = ce_loss(adv_logits_s1, plab_a.long()) + dice_loss(F.softmax(adv_logits_s1, dim=1), plab_a.unsqueeze(1))
            loss_adv_s2 = ce_loss(adv_logits_s2, plab_a.long()) + dice_loss(F.softmax(adv_logits_s2, dim=1), plab_a.unsqueeze(1))
            loss_adv = 0.5 * (loss_adv_s1 + loss_adv_s2)

            loss_rico = 0.0
            
            # --- FA-RC 胜负判断：用前景 hard Dice，并做滑动平均，降低 mini-batch 抖动 ---
            with torch.no_grad():
                out_l_s1_pure = model(img_a)
                out_l_s2_pure = model_rival(img_a)
                dice_s1 = hard_dice_score(out_l_s1_pure, lab_a, num_classes=num_classes)
                dice_s2 = hard_dice_score(out_l_s2_pure, lab_a, num_classes=num_classes)

                if winner_dice_ema_s1 is None:
                    winner_dice_ema_s1 = dice_s1
                    winner_dice_ema_s2 = dice_s2
                else:
                    m = args.winner_momentum
                    winner_dice_ema_s1 = m * winner_dice_ema_s1 + (1.0 - m) * dice_s1
                    winner_dice_ema_s2 = m * winner_dice_ema_s2 + (1.0 - m) * dice_s2

            if winner_dice_ema_s1 > winner_dice_ema_s2:  # S1 胜
                loss_rico = F.mse_loss(out_u_soft_s2, out_u_soft_s1.detach())
                s1_wins += 1
            else:  # S2 胜
                loss_rico = F.mse_loss(out_u_soft_s1, out_u_soft_s2.detach())
                s2_wins += 1

            # --- 总损失 ---
            loss = loss_s1 + loss_s2 + (rico_weight_ramped * loss_rico) + (adv_weight_ramped * loss_adv)
            
            loss_s1_history.append(loss_s1.item())
            loss_s2_history.append(loss_s2.item())

            optimizer.zero_grad()
            optimizer_rival.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            optimizer_rival.step()
            
            iter_num += 1
            
            update_ema_two_students(model, model_rival, ema_model, args.ema_decay)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_s1', loss_s1, iter_num)
            writer.add_scalar('info/loss_s2', loss_s2, iter_num)
            writer.add_scalar('info/loss_rico', loss_rico, iter_num)
            writer.add_scalar('info/loss_adv', loss_adv, iter_num)
            writer.add_scalar('info/rico_weight_ramped', rico_weight_ramped, iter_num)
            writer.add_scalar('info/adv_weight_ramped', adv_weight_ramped, iter_num)
            writer.add_scalar('info/mix_dice_s1', loss_dice_s1, iter_num)
            writer.add_scalar('info/mix_ce_s1', loss_ce_s1, iter_num)     
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num) 
            
            # <--- NEW: 记录难度变化 ---
            writer.add_scalar('info/bcp_ratio_square', ratio_sq, iter_num)
            writer.add_scalar('info/bcp_ratio_strip', ratio_st, iter_num)
            writer.add_scalar('info/winner_dice_batch_s1', dice_s1, iter_num)
            writer.add_scalar('info/winner_dice_batch_s2', dice_s2, iter_num)
            writer.add_scalar('info/winner_dice_ema_s1', winner_dice_ema_s1, iter_num)
            writer.add_scalar('info/winner_dice_ema_s2', winner_dice_ema_s2, iter_num)

            logging.info('iteration %d: loss: %f, loss_s1: %f, loss_s2: %f, rico: %f, adv: %f, r_sq: %.2f, r_st: %.2f'%(
                iter_num, loss.item(), loss_s1.item(), loss_s2.item(), loss_rico.item(), loss_adv.item(), ratio_sq, ratio_st
            ))
                
            if iter_num % 20 == 0:
                image = net_input_unl_s1[1, 0:1, :, :] 
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl_s1, dim=1), dim=1, keepdim=True) 
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l_s1[1, 0:1, :, :] 
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l_s1, dim=1), dim=1, keepdim=True) 
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            # --- 验证 S1, S2, Teacher ---
            if iter_num > 0 and iter_num % 200 == 0:
                
                # 1. 验证 model (S1)
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_model1_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_model1_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_model1_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : model_dice : %f' % (iter_num, performance))

                # 2. 验证 model_rival (S2)
                model_rival.eval()
                metric_list_2 = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i_2 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model_rival, classes=num_classes)
                    metric_list_2 += np.array(metric_i_2)
                
                metric_list_2 = metric_list_2 / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_model2_{}_dice'.format(class_i+1), metric_list_2[class_i, 0], iter_num)
                    writer.add_scalar('info/val_model2_{}_hd95'.format(class_i+1), metric_list_2[class_i, 1], iter_num)

                performance2 = np.mean(metric_list_2, axis=0)[0]
                writer.add_scalar('info/val_model2_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path_res = os.path.join(snapshot_path, 'iter_{}_dice_{}_res.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best_path_res = os.path.join(snapshot_path,'best_model_res.pth') 
                    torch.save(model_rival.state_dict(), save_mode_path_res)
                    torch.save(model_rival.state_dict(), save_best_path_res) 
                
                logging.info('iteration %d : model_rival_dice : %f' % (iter_num, performance2))

                # 3. 验证 ema_model (Teacher)
                ema_model.eval() 
                metric_list_ema = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i_ema = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], ema_model, classes=num_classes)
                    metric_list_ema += np.array(metric_i_ema)
                
                metric_list_ema = metric_list_ema / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_model_ema_{}_dice'.format(class_i+1), metric_list_ema[class_i, 0], iter_num)
                    writer.add_scalar('info/val_model_ema_{}_hd95'.format(class_i+1), metric_list_ema[class_i, 1], iter_num)

                performance_ema = np.mean(metric_list_ema, axis=0)[0]
                writer.add_scalar('info/val_model_ema_mean_dice', performance_ema, iter_num)

                # <--- PACL: 更新验证集性能，用于下一轮迭代的难度计算 --->
                latest_p_val = performance_ema

                if performance_ema > best_performance_ema:
                    best_performance_ema = performance_ema
                    save_best_path_ema = os.path.join(snapshot_path,'best_model_ema.pth') 
                    torch.save(ema_model.state_dict(), save_best_path_ema) 
                
                logging.info('iteration %d : ema_model_dice : %f' % (iter_num, performance_ema))
                
                # 恢复训练模式
                model.train()
                model_rival.train() 
                ema_model.train()

                # <--- NEW: 每 1000 iter 保存一次当前 loss 和验证集 Dice 到 TXT 文件 ---
                if iter_num % 1000 == 0:
                    with open(metrics_record_file, "a") as f:
                        # 记录 Iteration, 各项 Loss 值，以及三个模型的 Mean Dice
                        f.write(f"{iter_num},{loss.item():.6f},{loss_s1.item():.6f},{loss_s2.item():.6f},{loss_rico.item():.6f},{performance:.4f},{performance2:.4f},{performance_ema:.4f}\n")

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
            
    # <--- 最终可视化 ---
    logging.info('Training finished. Generating final visualizations...')
    
    # 1. 最终分割结果可视化
    final_best_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    if os.path.exists(final_best_model_path):
        logging.info(f"Loading best model 1 from {final_best_model_path} for final visualization.")
        model.load_state_dict(torch.load(final_best_model_path))
        model.eval()
        with torch.no_grad():
            try:
                val_sample_batch = next(iter(valloader))
                val_image_volume = val_sample_batch["image"].cuda().float()
                val_label_volume = val_sample_batch["label"].cuda()
                num_slices = val_image_volume.shape[1]
                slice_idx = num_slices // 2
                val_image = val_image_volume[:, slice_idx:slice_idx+1, :, :]
                val_label = val_label_volume[:, slice_idx:slice_idx+1, :, :]

                val_image_log = val_image[0, 0, ...].cpu().numpy()
                val_label_log = val_label[0, 0, ...].cpu().numpy() * 50

                pred_s1 = model(val_image)
                pred_s1_log = torch.argmax(torch.softmax(pred_s1, dim=1), dim=1, keepdim=False)[0, ...].cpu().numpy() * 50
                
                pred_s2_log = np.zeros_like(val_image_log) 
                final_best_model2_path = os.path.join(snapshot_path, 'best_model_res.pth')
                if os.path.exists(final_best_model2_path):
                    logging.info(f"Loading best model 2 from {final_best_model2_path} for final visualization.")
                    model_rival.load_state_dict(torch.load(final_best_model2_path))
                    model_rival.eval()
                    pred_s2 = model_rival(val_image)
                    pred_s2_log = torch.argmax(torch.softmax(pred_s2, dim=1), dim=1, keepdim=False)[0, ...].cpu().numpy() * 50

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                fig.suptitle(f'Final Validation Results (Best Models)')
                axes[0].imshow(val_image_log, cmap='gray')
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                axes[1].imshow(val_label_log, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_s1_log, cmap='gray')
                axes[2].set_title('Best Model 1 (S1) Pred.')
                axes[2].axis('off')

                axes[3].imshow(pred_s2_log, cmap='gray')
                axes[3].set_title('Best Model 2 (S2) Pred.')
                axes[3].axis('off')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                save_name = "Final_Segmentation_Results.png"
                fig.savefig(os.path.join(vis_save_path, save_name))
                writer.add_figure('val/Final_Segmentation_Results', fig, global_step=iter_num)
                plt.close(fig)
            except Exception as e:
                logging.error(f"Failed to generate final segmentation visualization: {e}")

    # 2. 整体损失曲线
    try:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
        def moving_average(a, n=100) : 
            if len(a) < n: return np.array([])
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
        
        ax_loss.plot(loss_s1_history, label='Loss S1 (raw)', alpha=0.3)
        ax_loss.plot(loss_s2_history, label='Loss S2 (raw)', alpha=0.3)
        ma_s1 = moving_average(loss_s1_history)
        ma_s2 = moving_average(loss_s2_history)
        if ma_s1.any(): ax_loss.plot(np.arange(100-1, len(loss_s1_history)), ma_s1, label='Loss S1 (MA-100)', color='blue')
        if ma_s2.any(): ax_loss.plot(np.arange(100-1, len(loss_s2_history)), ma_s2, label='Loss S2 (MA-100)', color='orange')

        ax_loss.set_title(f'Overall Student Losses')
        ax_loss.set_xlabel('Iteration Step')
        ax_loss.set_ylabel('Loss Value')
        ax_loss.legend()
        plt.grid(True)
        save_name_loss = "Final_Loss_Comparison.png"
        fig_loss.savefig(os.path.join(vis_save_path, save_name_loss))
        writer.add_figure('train/Final_Loss_Comparison', fig_loss, global_step=iter_num)
        plt.close(fig_loss)
    except Exception as e:
        logging.error(f"Failed to generate final loss curve visualization: {e}")

    # 3. 整体胜负统计
    try:
        fig_wins, ax_wins = plt.subplots(figsize=(6, 5))
        total_wins = s1_wins + s2_wins
        if total_wins > 0:
            s1_perc = (s1_wins / total_wins) * 100
            s2_perc = (s2_wins / total_wins) * 100
        else:
            s1_perc, s2_perc = 0, 0

        ax_wins.bar(['Student 1 Wins', 'Student 2 Wins'], [s1_wins, s2_wins], color=['cyan', 'magenta'])
        ax_wins.set_title(f'Overall RiCo Competition')
        ax_wins.set_ylabel('Total Win Count')
        if s1_wins > 0: ax_wins.text(0, s1_wins / 2, f'{s1_wins}\n({s1_perc:.1f}%)', ha='center', va='center', color='black')
        if s2_wins > 0: ax_wins.text(1, s2_wins / 2, f'{s2_wins}\n({s2_perc:.1f}%)', ha='center', va='center', color='black')

        save_name_wins = "Final_Win_Stats.png"
        fig_wins.savefig(os.path.join(vis_save_path, save_name_wins))
        writer.add_figure('train/Final_Win_Loss_Stats', fig_wins, global_step=iter_num)
        plt.close(fig_wins)
    except Exception as e:
        logging.error(f"Failed to generate final win stats visualization: {e}")
            
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = "/root/autodl-fs/DARC-main/code/model/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "/root/autodl-fs/DARC-main/code/model/ACDC_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    
    try:
        shutil.copy(__file__, self_snapshot_path)
    except NameError:
        pass 
        #Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)