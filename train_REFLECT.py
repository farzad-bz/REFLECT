
import torch
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
        
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = f'{args.model_size}-{args.modality}'
        args.experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
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
    return logger



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    
    """
    Trains a new  model.
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
    
    logger = create_logger_and_dirs(args)
    
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
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        scheduler.step()
        
        if epoch % args.ckpt_every == 0 and epoch>0:
            
            if rank == 0: 
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
            
    
    logger.info("Done!")
    cleanup()

if __name__ == "__main__":

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
    args.results_dir = f"./results_Medical-REFLECT_{args.dataset}_{args.model_size}_{args.modality}_{args.image_size}_{args.vae}"
    main(args)
