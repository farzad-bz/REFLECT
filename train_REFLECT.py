# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import json
from time import time
from PIL import Image
from copy import deepcopy
from glob import glob
import argparse
import logging
import os
import random
import yaml
import torch.nn.functional as F

from anomalib import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from medical_models import UNET_models
from transformers import get_cosine_schedule_with_warmup
from MedicalDataLoader import BraTS2021Dataset, ATLASDataset
from huggingface_hub import hf_hub_download
import wandb


import torch.nn as nn


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger_and_dirs(args):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:

        wandb_logger = wandb.init(entity='beizaeefarzad-cole-de-technologie-sup-rieure', project='RFAD-Medical', config=args, tags=['UNet_L', 'BraTS', args.model_size ,args.modality, args.vae],
                    notes=f'RFAD_{args.vae}_{args.modality}_{args.model_size}')
        
        args.wand_run_id = wandb.run.id
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = f'{args.model_size}-{args.modality}'
        args.experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.wand_run_id}"  # Create an experiment folder
        args.checkpoint_dir = f"{args.experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.experiment_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        wandb_logger = None
    return logger, wandb_logger



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    
    job_starting_time = time()
    new_job_submitted = False
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    logger, wandb_logger = create_logger_and_dirs(args)
    
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup an experiment folder:
    if rank == 0:
        
        if not os.path.exists(f'{args.results_dir}/last_epoch.txt'):
            with open(f'{args.results_dir}/last_epoch.txt', 'w') as f:
                f.write('0')
        
        with open(f'{args.results_dir}/last_epoch.txt', 'r') as f:
            args.last_trained_epoch = int(f.read().strip())
            if args.last_trained_epoch != 0:
                args.last_trained_epoch += 1
        
            
        with open(f'{args.experiment_dir}/args.yml', 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)   

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    dictss = {}
    if args.vae == 'kl_f4':
        in_channels = out_channels = 3
    else:
        in_channels = out_channels = 4
        
    model = UNET_models[args.model_size](in_channels=in_channels, out_channels=out_channels)
    if args.last_trained_epoch != 0 :
        try:
            ckpt = sorted(glob(f'{args.results_dir}/*/checkpoints/best_ap.pt'))[-1]
        except:
            ckpt = sorted(glob(f'{args.results_dir}/*/*/checkpoints/best_ap.pt'))[-1]
        state_dict = torch.load(ckpt)['model']
        logger.info(model.load_state_dict(state_dict))
        
    model = DDP(model.to(device), device_ids=[rank])

    if args.vae == 'kl_f4':
        vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf4.pt")
        vae = torch.load(vae_model_path)
        embedding_dim = 3
        compression_factor = 4


    elif args.vae == 'kl_f8':
        vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf8.pt")
        vae = torch.load(vae_model_path)
        embedding_dim = 4
        compression_factor = 8
        
    vae.eval()
    vae.to(device)
    
    logger.info(f"model Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    
    if args.dataset == 'BraTS2021':
        dataset = BraTS2021Dataset('train', transform=transform, image_size=args.image_size, augment=args.augmentation, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor, dtd_emb_dir=f'{args.data_dir}/dtd_embeddings_medical_{args.vae}.npy')
        loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, num_workers=4, drop_last=False)

        val_dataset = BraTS2021Dataset('test', transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor, dtd_emb_dir=f'{args.data_dir}/dtd_embeddings_medical_{args.vae}.npy')
        val_loader = DataLoader(val_dataset, batch_size=args.global_batch_size, shuffle=False, num_workers=4, drop_last=False)
    else:
        dataset = ATLASDataset('train', transform=transform, image_size=args.image_size, augment=args.augmentation, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor, dtd_emb_dir=f'{args.data_dir}/dtd_embeddings_medical_{args.vae}.npy')
        loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, num_workers=4, drop_last=False)

        val_dataset = ATLASDataset('test', transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor, dtd_emb_dir=f'{args.data_dir}/dtd_embeddings_medical_{args.vae}.npy')
        val_loader = DataLoader(val_dataset, batch_size=args.global_batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    
    if args.global_batch_size < 64:
        accumulation_steps = 4
    else:
        accumulation_steps = 2 
    
    logger.info(f"Dataset contains {len(dataset):,} training images and  {len(val_dataset):,} validation images")

    adjusted_epochs = args.epochs

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=args.warmup_epochs,
        num_training_steps=args.epochs*2,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adjusted_epochs, eta_min=args.lr/100)

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {adjusted_epochs} epochs...")
    
    for epoch in range(args.last_trained_epoch):
        scheduler.step()
    for epoch in range(args.last_trained_epoch, adjusted_epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for ii, (x, replacement, mask) in enumerate(loader):
            x = x.to(device)
            replacement = replacement.to(device)
            mask = mask.to(device)

            t = torch.rand(x.size(0), 1).to(torch.float32)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                z1 = vae.encode(x).sample().mul_(0.18215)
                z_0 = torch.sqrt((1-mask)) * z1 + torch.sqrt(mask) * replacement
                target_velocity = z1 - z_0
                z_t = z_0 + t.view(-1, 1, 1, 1).to(device) * target_velocity
                
            pred_velocity = model(z_t, t)  # Flatten spatial dimensions
        
            loss = F.mse_loss(pred_velocity, target_velocity)
            loss.backward()
            
            if (ii + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad() 

            # Log loss values:
            running_loss += loss.item()
            # running_mt += loss_dict["mt_prediction"].mean().item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # if rank == 0: 
                #     if (not new_job_submitted) and ((time() - job_starting_time) > 10000):
                #         new_job_submitted = True 
                #         os.system(f'sbatch train_job_{args.dataset}_{args.center_size}.sh')
                
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}) MSE Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank == 0: 
                    wandb_logger.log({'Train MSE Loss': avg_loss})
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
        scheduler.step()
        
        if epoch % args.ckpt_every == 0 and epoch>0:
            
            if rank == 0: 
                wandb_logger.log({'Validation auroc': results['auroc'],
                    'Validation f1_max': results['f1_max'],
                    'Validation ap': results['ap'],
                    'Validation aurocsp': results['aurocsp'],
                    'Validation apsp': results['apsp'],
                    'Validation f1sp': results['f1sp'],
                    })
                # Save checkpoint:
                checkpoint = {
                    "model": model.module.state_dict(),
                    # "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{args.checkpoint_dir}/last.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                with open(f'{args.results_dir}/last_epoch.txt', 'w') as f:
                    f.write(str(epoch))

            dist.barrier()
            

    model.eval()  # important! This disables randomized embedding dropout
    results = evaluation(model, vae, val_loader, args, logger, device=device, rank=rank)
    print(results)
    
    logger.info("Done!")
    cleanup()
    
    
    

def calculate_metrics(ground_truth, prediction, device='cuda'):
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    

    # auprc = metrics.AUPR()
    # auprc_score = auprc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    # aupro = metrics.AUPRO(fpr_limit=0.3)
    
    auroc = metrics.AUROC()
    auroc_score = auroc(torch.from_numpy(flat_pred).to(device), torch.from_numpy(flat_gt.astype(int)).to(device)).cpu().numpy()

    f1max = metrics.F1Max()
    f1_max_score = f1max(torch.from_numpy(flat_pred).to(device), torch.from_numpy(flat_gt.astype(int)).to(device)).cpu().numpy()
    
    # # Dice Coefficient (same as F1 score for binary)
    # flat_gt = ground_truth.flatten()
    # flat_pred = prediction.flatten().round()
    # intersection = np.sum(flat_gt * flat_pred)
    # dice = (2. * intersection) / (np.sum(flat_gt) + np.sum(flat_pred))
    
    ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    gt_list_sp = []
    pr_list_sp = []
    for idx in range(len(ground_truth)):
        gt_list_sp.append(np.max(ground_truth[idx]))
        sp_score = prediction[idx].max()
            # else:
            #     anomaly_map = anomaly_map.ravel()
            #     sp_score = np.sort(anomaly_map)[-int(anomaly_map.shape[0] * max_ratio):]
            #     sp_score = sp_score.mean()
        pr_list_sp.append(sp_score)

    gt_list_sp = np.array(gt_list_sp).astype(np.int32)
    pr_list_sp = np.array(pr_list_sp)

    # apsp = average_precision_score(gt_list_sp, pr_list_sp)
    aurocsp = auroc(torch.from_numpy(pr_list_sp).to(device), torch.from_numpy(gt_list_sp).to(device)).cpu().numpy()
    # f1sp = f1max(torch.from_numpy(pr_list_sp).to(device), torch.from_numpy(gt_list_sp).to(device)).cpu().numpy()
    
    return auroc_score ,f1_max_score, ap, aurocsp, 0, 0    

    
def evaluation(model, vae, val_loader, args, logger, device='cuda', rank=0, inference_steps=5):
    dist.barrier()

    auroc_score_s = []
    f1_max_score_s = []
    ap_s = []
    aurocsp_s = []
    apsp_s = []
    f1sp_s = []

    model.eval()

    segmentation_s = []
    encoded_s = []
    latent_samples_s = []
    # latent_size = 32

    
    for ii, (x, seg) in enumerate(val_loader):
    # if ii%2==0:
    #     continue
    # x = x.repeat([num_iteration,1,1,1]).to(device)

        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            encoded = vae.encode(x.to(device)).mean.mul_(0.18215)
            
            latent_sample = encoded.clone()
        # Euler solver (can use higher-order methods)
            dt = 1 / inference_steps  # Step size (adjust for accuracy/speed tradeoff)
            for time in torch.arange(0, 1, dt):
                t = time * torch.ones((encoded.shape[0], 1)).to(torch.float32).to(device)
                # velocity = model(encoded, t)
                velocity = model(latent_sample, t)
                latent_sample = latent_sample + velocity * dt
            

        segmentation_s += [_seg.unsqueeze(0) for _seg in seg]
        encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
        latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_sample]
    pr = []   
    gt = []
    for segmentation, encoded, latent_sample in zip(segmentation_s, encoded_s, latent_samples_s):
        latent_difference = (((((torch.abs(latent_sample-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        # latent_difference = (np.clip(latent_difference, 0.0 , 0.4)) * 2.5
        latent_difference = smooth_mask(latent_difference, sigma=1)
        latent_difference = resize(latent_difference, (args.image_size, args.image_size))
        pr.append(latent_difference)

        gt.append(segmentation[0,:,:].cpu().numpy())
    pr = np.stack(pr, axis=0)
    gt = np.stack(gt, axis=0)
    gt = (gt>0).astype(np.int32)
    auroc_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(gt, pr, device=device)

    auroc_score_s.append(np.round(auroc_score,4))
    f1_max_score_s.append(np.round(f1_max_score,4))
    ap_s.append(np.round(ap,4))
    aurocsp_s.append(np.round(aurocsp,4))
    apsp_s.append(np.round(apsp,4))
    f1sp_s.append(np.round(f1sp,4))
    
    results = {'auroc':np.mean(auroc_score_s),'f1_max':np.mean(f1_max_score_s), 'ap':np.mean(ap_s), 'aurocsp':np.mean(aurocsp_s), 'apsp':np.mean(apsp_s), 'f1sp':np.mean(f1sp_s)}
    if torch.distributed.get_rank() == 0:
        logger.info(f'Validation: {results}')
    model.train()
    dist.barrier()
    return results

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['BraTS2021', 'ATLAS2'], default="BraTS2021")
    parser.add_argument("--model_size", type=str, choices=['UNet_XS', 'UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default="UNet_L")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--modality", type=str, choices=['t1', 't1ce', 't2', 'flair'], default="t1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--global-batch-size", type=int, default=96)
    parser.add_argument("--global-seed", type=int, default=10)
    parser.add_argument("--vae", type=str, choices=["kl_f8", "kl_f4"], default="kl_f8")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--data_dir", type=bool, default='.')
    parser.add_argument("--augmentation", type=bool, default=False)
    parser.add_argument("--max-objects", type=int, default=4)

    args = parser.parse_args()
    args.last_trained_epoch = 0
    args.results_dir = f"./results_Medical-REFLECT_{args.dataset}_{args.model_size}_{args.modality}_{args.image_size}_{args.vae}_with_augmentation"
    main(args)
