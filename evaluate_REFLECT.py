
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
import yaml
from scipy.ndimage import gaussian_filter
from medical_models import UNET_models
from MedicalDataLoader import BraTS2021Dataset, ATLASDataset
from huggingface_hub import hf_hub_download
from anomalib import metrics
from sklearn.metrics import average_precision_score
import cv2
from PIL import Image
from skimage.transform import resize
import copy


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


def compute_dice(anomaly_map, segmentation, th):
    anomaly_map[anomaly_map>th]=1
    anomaly_map[anomaly_map<1]=0
    if sum(segmentation.flatten())==0:
        if sum(anomaly_map.flatten())==0:
            return 1
        else:
            return 0
    
    eps = 1e-6
    # flatten label and prediction tensors
    inputs = anomaly_map.flatten()
    targets = segmentation.flatten()

    intersection = (inputs * targets).sum()
    dice = (2. * intersection) / (inputs.sum() + targets.sum() + eps)
    return dice


def dsc_max(anomaly_maps, segmentations):
    dice_scores = []
    ths = np.linspace(0, 1, 101)
    best_dsc = 0
    for dice_threshold in ths:
        dice_scores = []
        for k in range(len(anomaly_maps)):
            dice = compute_dice(copy.deepcopy(np.asarray(anomaly_maps[k]).flatten()), copy.deepcopy(np.asarray(segmentations[k]).flatten()), dice_threshold)
            dice_scores.append(dice)
        if np.mean(dice_scores) > best_dsc:
            best_dsc = np.mean(dice_scores)
    return best_dsc 
    
    
def calculate_metrics(ground_truth, prediction):
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    
    max_dicescore = dsc_max(prediction, ground_truth)
    
    auroc = metrics.AUROC()
    auroc_score = auroc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    f1max = metrics.F1Max()
    f1_max_score = f1max(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))
    
    ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    return auroc_score.cpu().numpy() ,f1_max_score.cpu().numpy(), ap, max_dicescore


def visualize(anomaly_maps, segmentations, xs,  image_samples, args):
        counter = -1
        os.makedirs(os.path.join(args.parent_dir, f'visualization/{args.backward_steps}_backward_steps/'), exist_ok=True)
        for anomaly_map, segmentation, x,  image_samples in zip(anomaly_maps, segmentations, xs,  image_samples):
                counter+=1
                visualization_image = np.zeros((4*args.image_size, args.image_size, 3)).astype(np.uint8)
                input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
                output_image = ((np.clip(image_samples[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)

                scoremap = cv2.applyColorMap((anomaly_map*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
                anomal_map_img = (0.5 * input_image + (1 - 0.5) * scoremap).astype(np.uint8)
                visualization_image[:args.image_size, :] = input_image
                visualization_image[args.image_size:2*args.image_size, :] = output_image
                visualization_image[2*args.image_size:3*args.image_size, :] = np.repeat(segmentation.cpu().numpy(), 3, axis=0).transpose([1,2,0])
                visualization_image[3*args.image_size:, :] = anomal_map_img
                Image.fromarray(visualization_image).save(os.path.join(args.parent_dir, f'visualization/{args.backward_steps}_backward_steps/{counter}.png'))


def evaluate(x0s, segmentations, encodeds,  image_samples, latent_samples, args):
        anomaly_maps = []
        gt = []

        for x, segmentation, encoded, image_sample, latent_sample in zip(x0s, segmentations, encodeds,  image_samples, latent_samples):
                image_difference = (((((torch.abs(image_sample-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
                image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5

                image_difference = smooth_mask(image_difference, sigma=3)
                
                latent_difference = (((((torch.abs(latent_sample-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
                latent_difference = (np.clip(latent_difference, 0.0 , 0.4)) * 2.5
                latent_difference = smooth_mask(latent_difference, sigma=1)
                latent_difference = resize(latent_difference, (256, 256))
                
                final_anomaly = 1/2*image_difference + 1/2*latent_difference
                anomaly_maps.append(final_anomaly)
                gt.append(segmentation[0,:,:].cpu().numpy())
                
        anomaly_maps = np.stack(anomaly_maps, axis=0)
        gt = np.stack(gt, axis=0)
        gt = (gt>0).astype(np.int32)

        auroc_score ,f1_max_score, ap, max_dicescore = calculate_metrics(gt, anomaly_maps)
        with open(os.path.join(args.parent_dir, f'results_with_{args.backward_steps}_backward_steps.txt'), 'w') as f:
            f.write('max Dice score:{:.4f}\nGlobal max Dice score: {:.4f}\nAUROC: {:.4f}\nAP: {:.4f}'.format(
                np.round(max_dicescore, 4),
                np.round(f1_max_score, 4),
                np.round(auroc_score, 4),
                np.round(ap,4)
            ))
            
        return anomaly_maps, {'max Dice score':np.round(max_dicescore, 4), 'Global max Dice score': np.round(f1_max_score, 4), 'AUROC':np.round(auroc_score, 4), 'AP':np.round(ap,4)}




def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    if args.vae == 'kl_f4':
        in_channels = out_channels = 3
    else:
        in_channels = out_channels = 4
        
    model = UNET_models[args.model](in_channels=in_channels, out_channels=out_channels)
    try:
        state_dict = torch.load(args.model_path)['model']
        print(model.load_state_dict(state_dict))
    except:
        raise Exception('Provided trained model path could not be found or it is not consistent with model params.')
    
    model.eval()
    model.to(device)    

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
    
    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    
    if args.dataset == 'BraTS':
        test_dataset = BraTS2021Dataset('test', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    else:
        test_dataset = ATLASDataset('test', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    
    print(f"Dataset contains {len(test_dataset)} Test images.")

    x_s = []
    encoded_s = []
    image_samples_s = []
    latent_samples_s = []
    x0_s = []
    segmentation_s = []
    
    print('=-='*20)
    print('Starting evaluation...')
    print('=-='*20)
    for ii, (x, mask, seg) in enumerate(test_loader):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            encoded = vae.encode(x.to(device)).mean.mul_(0.18215)#Normalization params got from LDM package
            
            latent_sample = encoded.clone()
        # Euler solver (can use higher-order methods)
            dt = 1 / args.backward_steps   # Step size (adjust for accuracy/speed tradeoff)
            for time in torch.arange(0, 1, dt):
                t = time * torch.ones((encoded.shape[0], 1)).to(torch.float32).to(device)
                # velocity = model(encoded, t)
                velocity = model(latent_sample, t)
                latent_sample = latent_sample + velocity * dt

            image_samples = vae.decode(latent_sample / 0.18215) #* (1-mask)
            x0 = vae.decode(encoded / 0.18215) #* (1-mask)

            x_s += [_x.unsqueeze(0) for _x in x]
            segmentation_s += [_seg.unsqueeze(0) for _seg in seg]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_sample]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]

    anomaly_maps, results = evaluate(x0_s, segmentation_s, encoded_s,  image_samples_s, latent_samples_s, args)
    for key, val in results.items():
        print(key, ' : ', val)
    visualize(anomaly_maps, segmentation_s, x0_s, image_samples_s, args)
    print('=-='*20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default='.')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--backward-steps", type=int, default=1)
    

    args = parser.parse_args()
    
    args.parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.model_path)))
    try:
        with open(os.path.join(args.parent_dir, 'args.yml'), 'r') as file:
            config = yaml.safe_load(file)  
        args.dataset = config['dataset']
        args.image_size = int(config['image_size'])
        args.modality = config['modality']
        args.vae = config['vae']
        args.model = config['model']          
    except:
        raise Exception("YAML config file could not be found in the parent folder of the provided model path")

    main(args)
    



