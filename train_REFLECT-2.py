
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from time import time
from copy import deepcopy
from glob import glob
import argparse
import logging
import os
import yaml
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter
from medical_models import UNET_models
from transformers import get_cosine_schedule_with_warmup
from MedicalDataLoader import BraTS2021Dataset, ATLASDataset
from huggingface_hub import hf_hub_download



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
        model_string_name = f'{args.model}-{args.modality}'
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
    Trains a new model.
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
        with open(f'{args.experiment_dir}/args.yml', 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)   

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.vae == 'kl_f4':
        in_channels = out_channels = 3
    else:
        in_channels = out_channels = 4
        
    model = UNET_models[args.model](in_channels=in_channels, out_channels=out_channels)

    try:
        ckpt = args.REFLECT_1_path
        state_dict = torch.load(ckpt)['model']
    except:
        raise Exception('REFLECT-1 trained model could not be found or it is not consistent with model params.')
    
    logger.info(model.load_state_dict(state_dict))
    logger.info('First velocity model loaded')
        

    first_velocity_model = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(first_velocity_model, False)

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
    
    if args.dataset == 'BraTS':
        dataset = BraTS2021Dataset('train', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=args.augmentation, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor, dtd_emb_dir=os.path.join(args.dtd_dir, f'dtd_embeddings_medical_{args.vae}.npy'))
        loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, num_workers=4, drop_last=False)
    else:
        dataset = ATLASDataset('train', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=args.augmentation, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor, dtd_emb_dir=os.path.join(args.dtd_dir, f'dtd_embeddings_medical_{args.vae}.npy'))
        loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, num_workers=4, drop_last=False)
        
        
    if args.global_batch_size < 64:
        accumulation_steps = 4
    else:
        accumulation_steps = 2 
    
    logger.info(f"Dataset contains {len(dataset):,} training images.")

    adjusted_epochs = args.epochs

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=args.warmup_epochs,
        num_training_steps=args.epochs*2,
    )

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    first_velocity_model.eval()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {adjusted_epochs} epochs...")
    
    for epoch in range(adjusted_epochs):
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
                
                z1_2 = deepcopy(z_0)
                dt = 1 / args.backward_steps  # Step size (adjust for accuracy/speed tradeoff)
                for tt in torch.arange(0, 1, dt):
                    ttt = tt * torch.ones((z1_2.shape[0], 1)).to(torch.float32).to(device)
                    # velocity = model(encoded, t)
                    velocity = first_velocity_model(z1_2, ttt)
                    z1_2 = z1_2 + velocity * dt
                
                target_velocity = z1_2 - z_0
                z_t = z_0 + t.view(-1, 1, 1, 1).to(device) * target_velocity
                
            pred_velocity = model(z_t, t)  # Flatten spatial dimensions
        
            loss = F.mse_loss(pred_velocity, target_velocity)
            loss.backward()
            
            if (ii + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad() 

            # Log loss values:
            running_loss += loss.item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
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


    logger.info("Done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=96)
    parser.add_argument("--global-seed", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default='.')
    parser.add_argument("--dtd-dir", type=str, default='.')
    parser.add_argument("--REFLECT-1-path", type=str, default='.')
    parser.add_argument("--augmentation", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--backward-steps", type=int, default=5)
    

    args = parser.parse_args()
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.REFLECT_1_path))), 'args.yml'), 'r') as file:
            config = yaml.safe_load(file)  
        args.dataset = config['dataset']
        args.image_size = int(config['image_size'])
        args.modality = config['modality']
        args.vae = config['vae']
        args.model = config['model']          
    except:
        raise Exception("YAML config file could not be found in the parent folder of REFLECT-1-path")



    args.results_dir = f"./REFLECT-2_{args.dataset}_{args.model}_{args.modality}_{args.image_size}_{args.vae}"
    main(args)
