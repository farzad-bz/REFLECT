from operator import index
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
# import torchio as tio
import warnings
from glob import glob
import albumentations as A
import random
import cv2

warnings.filterwarnings("ignore")

class BraTS2021Dataset(Dataset):
    """ABIDE dataset."""

    def __init__(self, mode, rootdir= './Brats2021-slices', modality="T1", dtd_emb_dir='./dtd_embeddings.npy', embedding_dim=4, compression_factor=8, max_objects=4, transform=None,  normal=True, image_size=256, augment=True):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.mode = mode
        if mode == 'train' and normal==False:
            raise Exception('training data should be normal')
        self.augment = augment
        self.normal = normal
        self.embedding_dim = embedding_dim
        self.transform = transform
        self.image_size = image_size
        self.compression_factor = compression_factor
        self.embedding_size = image_size // compression_factor 
        self.max_objects = max_objects
        if mode=='train':
            self.dtd_embeddings = np.load(dtd_emb_dir)
        else:
            self.dtd_embeddings = None
        
        self.image_paths = glob(os.path.join(rootdir, mode, f'*-{modality}.png'))
        self.mask_paths = [path.replace(f'-{modality}', '-brainmask') for path  in self.image_paths]
        
        if mode=='val' or mode=='test':
            self.seg_paths = [path.replace(f'-{modality}', '-segmentation') for path  in self.image_paths]
              
        if self.augment:
           self.aug = A.Compose([
                A.Affine (translate_px=int(self.image_size//32), p=0.5),
                A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5),])

    def transform_volume(self, x):
        x = torch.from_numpy(x.transpose((-1, 0 , 1)))
        return x

    def __len__(self):
        return len(self.image_paths)
        

    def __getitem__(self, index):
        img = np.array(ImageOps.pad(Image.open(self.image_paths[index]), (256,256), color="#000").convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        brain_mask = np.array(ImageOps.pad(Image.open(self.mask_paths[index]), (256,256), color="#000").convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        # img = np.array(Image.open(self.image_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        # brain_mask = np.array(Image.open(self.mask_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        
        img = img.astype(np.uint8)
        brain_mask = (brain_mask>0.0).astype(np.int32)
        if self.augment and self.mode=='train':
            augmented = self.aug(image=img, mask=brain_mask)
            img = augmented['image']
            brain_mask = augmented['mask']
            img[brain_mask==0] = 0
                
                
        img = img.astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(img)
        else:
            img = self.transform_volume(img)
            img = (img-0.5)/0.5
            
            
        if self.mode == 'train':
            initial_mask = (cv2.resize((brain_mask*255).astype(np.uint8), (self.embedding_size,self.embedding_size))>0).astype(np.int32)
            mask = self.generate_random_walk_contour(initial_mask, self.max_objects)
            replacement = np.zeros((self.embedding_dim, self.embedding_size, self.embedding_size)).astype(np.float32)
            for j in range(1, mask.max()+1):
                texture_img = self.dtd_embeddings[np.random.randint(self.dtd_embeddings.shape[0])]
                try:
                    x_offset = np.random.randint(texture_img.shape[1]-self.embedding_size)
                    y_offset = np.random.randint(texture_img.shape[2]-self.embedding_size)
                except:
                    x_offset = 0
                    y_offset = 0
                texture_img = texture_img[:self.embedding_dim ,x_offset:x_offset+self.embedding_size, y_offset:y_offset+self.embedding_size]

                if random.random()<0.5:
                    if (mask==(j)).any():
                        replacement[mask==(j)] = texture_img[mask==(j)]
                else:
                    alpha = np.random.random()
                    noise_base = np.random.randn(self.embedding_dim).reshape(self.embedding_dim, 1, 1)
                    noise_additive = np.random.randn(self.embedding_dim, self.embedding_size, self.embedding_size).astype(np.float32)
                    noise = np.sqrt(alpha)*noise_base + np.sqrt(1-alpha)*noise_additive
                    # print('mask', mask.shape)
                    # print('noise', noise.shape)
                    # print('replacement', replacement.shape)
                    if (mask==(j)).any():
                        # print(mask==(j)))
                        replacement[mask==(j)] = noise[mask==(j)]
                
            mask = mask.astype(np.float32)
            for j in range(1, int(mask.max())+1):
                mask[mask==j] = np.random.uniform(0.2, 1)
                    
            replacement = torch.from_numpy(replacement)
            mask = torch.from_numpy((mask).astype(np.float32))
            return img, replacement, mask
        else:
            seg = np.array(Image.open(self.seg_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
            return img, mask, seg.astype(np.float32)
            
   

    def generate_random_walk_contour(self, initial_mask, max_objects=4):
        
        steps_adj = (self.embedding_size/32)**2
        
        # Determine the number of objects (probability inversely proportional to number)
        num_objects = np.random.choice(
            np.arange(1, max_objects + 1), 
            1, 
            p=(1/np.arange(1, max_objects+1)) / np.sum(1/np.arange(1, max_objects+1))
        ).item()

        # Set number of steps based on number of objects
        steps = {1:int(200 * steps_adj), 2:int(120 * steps_adj), 3:int(100 * steps_adj), 4:int(80 * steps_adj)}.get(num_objects, 100)

        masks = []
        for i in range(num_objects):
            mask = np.zeros((self.embedding_size, self.embedding_size), dtype=np.uint8)
            
            # Get all valid starting positions from initial_mask
            y_indices, x_indices = np.where(initial_mask != 0)
            possible_starts = list(zip(x_indices, y_indices))
            
            # Handle edge case where initial_mask is empty
            x, y = possible_starts[np.random.choice(len(possible_starts))]
            
            # Random walk parameters
            points = [(x, y)]
            weights = [10 - 9/steps * x for x in np.arange(steps)]
            num_steps = random.choices(np.arange(steps), weights=weights, k=1)[0]
            
            # Perform random walk
            for _ in range(num_steps):
                dx, dy = np.random.choice([-1, 0, 1], size=2)
                x_, y_ = np.clip(x + dx, 0, self.embedding_size-1), np.clip(y + dy, 0, self.embedding_size-1)
                if (x_, y_) in possible_starts:
                    x, y = x_, y_
                    points.append((x, y))
            
            # Create contour from points
            contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [contour], 255)
            
            # Apply Gaussian blur
            kernel_size = 1 if num_steps < 10 else 3
            # mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 1)
            mask = (mask > 128).astype(np.uint8)
            masks.append(mask)

        # Combine masks with object IDs
        final_mask = np.zeros((self.embedding_size, self.embedding_size), dtype=np.uint8)
        for i, obj_mask in enumerate(masks, 1):
            final_mask = np.where(final_mask == 0, obj_mask * i, final_mask)
            
    
        if random.random()<0.9:
            return np.repeat(np.expand_dims(final_mask, axis=0), self.embedding_dim, axis=0)
        else:
            return np.zeros((self.embedding_dim, self.embedding_size, self.embedding_size), np.uint8)
        
        
        
        
        
        
        
   
   

class ATLASDataset(Dataset):
    """ATLAS dataset."""

    def __init__(self, mode, rootdir= './ATLAS-slices', modality="T1", dtd_emb_dir='./dtd_embeddings.npy', embedding_dim=4, compression_factor=8, max_objects=4, transform=None,  normal=True, image_size=256, augment=True):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.mode = mode
        if mode == 'train' and normal==False:
            raise Exception('training data should be normal')
        self.augment = augment
        self.normal = normal
        self.embedding_dim = embedding_dim
        self.transform = transform
        self.image_size = image_size
        self.compression_factor = compression_factor
        self.embedding_size = image_size // compression_factor 
        self.max_objects = max_objects
        if mode=='train':
            self.dtd_embeddings = np.load(dtd_emb_dir)
        else:
            self.dtd_embeddings = None
        
        self.image_paths = glob(os.path.join(rootdir, mode, f'*-{modality}.png'))
        self.mask_paths = [path.replace(f'-{modality}', '-brainmask') for path  in self.image_paths]
        
        if mode=='val' or mode=='test':
            self.seg_paths = [path.replace(f'-{modality}', '-segmentation') for path  in self.image_paths]
              
        if self.augment:
           self.aug = A.Compose([
                A.Affine (translate_px=int(self.image_size//32), p=0.5),
                A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5),])

    def transform_volume(self, x):
        x = torch.from_numpy(x.transpose((-1, 0 , 1)))
        return x

    def __len__(self):
        return len(self.image_paths)
        

    def __getitem__(self, index):
        img = np.array(ImageOps.pad(Image.open(self.image_paths[index]), (256,256), color="#000").convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        brain_mask = np.array(ImageOps.pad(Image.open(self.mask_paths[index]), (256,256), color="#000").convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        # img = np.array(Image.open(self.image_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        # brain_mask = np.array(Image.open(self.mask_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
        
        img = img.astype(np.uint8)
        brain_mask = (brain_mask>0.0).astype(np.int32)
        if self.augment and self.mode=='train':
            augmented = self.aug(image=img, mask=brain_mask)
            img = augmented['image']
            brain_mask = augmented['mask']
            img[brain_mask==0] = 0
                
                
        img = img.astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(img)
        else:
            img = self.transform_volume(img)
            img = (img-0.5)/0.5
            
            
        if self.mode == 'train':
            initial_mask = (cv2.resize((brain_mask*255).astype(np.uint8), (self.embedding_size,self.embedding_size))>0).astype(np.int32)
            mask = self.generate_random_walk_contour(initial_mask, self.max_objects)
            replacement = np.zeros((self.embedding_dim, self.embedding_size, self.embedding_size)).astype(np.float32)
            for j in range(1, mask.max()+1):
                texture_img = self.dtd_embeddings[np.random.randint(self.dtd_embeddings.shape[0])]
                try:
                    x_offset = np.random.randint(texture_img.shape[1]-self.embedding_size)
                    y_offset = np.random.randint(texture_img.shape[2]-self.embedding_size)
                except:
                    x_offset = 0
                    y_offset = 0
                texture_img = texture_img[:self.embedding_dim ,x_offset:x_offset+self.embedding_size, y_offset:y_offset+self.embedding_size]

                if random.random()<0.5:
                    if (mask==(j)).any():
                        replacement[mask==(j)] = texture_img[mask==(j)]
                else:
                    alpha = np.random.random()
                    noise_base = np.random.randn(self.embedding_dim).reshape(self.embedding_dim, 1, 1)
                    noise_additive = np.random.randn(self.embedding_dim, self.embedding_size, self.embedding_size).astype(np.float32)
                    noise = np.sqrt(alpha)*noise_base + np.sqrt(1-alpha)*noise_additive
                    if (mask==(j)).any():
                        # print(mask==(j)))
                        replacement[mask==(j)] = noise[mask==(j)]
                
            mask = mask.astype(np.float32)
            for j in range(1, int(mask.max())+1):
                mask[mask==j] = np.random.uniform(0.2, 1)
                    
            replacement = torch.from_numpy(replacement)
            mask = torch.from_numpy((mask).astype(np.float32))
            return img, replacement, mask
        else:
            seg = np.array(Image.open(self.seg_paths[index]).convert('L').resize((self.image_size, self.image_size))).astype(np.uint8)
            return img, mask, seg.astype(np.float32)
            
            
   
   
            

    def generate_random_walk_contour(self, initial_mask, max_objects=4):
        
        steps_adj = (self.embedding_size/32)**2
        
        # Determine the number of objects (probability inversely proportional to number)
        num_objects = np.random.choice(
            np.arange(1, max_objects + 1), 
            1, 
            p=(1/np.arange(1, max_objects+1)) / np.sum(1/np.arange(1, max_objects+1))
        ).item()

        # Set number of steps based on number of objects
        steps = {1:int(200 * steps_adj), 2:int(120 * steps_adj), 3:int(100 * steps_adj), 4:int(80 * steps_adj)}.get(num_objects, 100)

        masks = []
        for i in range(num_objects):
            mask = np.zeros((self.embedding_size, self.embedding_size), dtype=np.uint8)
            
            # Get all valid starting positions from initial_mask
            y_indices, x_indices = np.where(initial_mask != 0)
            possible_starts = list(zip(x_indices, y_indices))
            
            # Handle edge case where initial_mask is empty
            x, y = possible_starts[np.random.choice(len(possible_starts))]
            
            # Random walk parameters
            points = [(x, y)]
            weights = [10 - 9/steps * x for x in np.arange(steps)]
            num_steps = random.choices(np.arange(steps), weights=weights, k=1)[0]
            
            # Perform random walk
            for _ in range(num_steps):
                dx, dy = np.random.choice([-1, 0, 1], size=2)
                x_, y_ = np.clip(x + dx, 0, self.embedding_size-1), np.clip(y + dy, 0, self.embedding_size-1)
                if (x_, y_) in possible_starts:
                    x, y = x_, y_
                    points.append((x, y))
            
            # Create contour from points
            contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [contour], 255)
            
            # Apply Gaussian blur
            kernel_size = 1 if num_steps < 10 else 3
            # mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 1)
            mask = (mask > 128).astype(np.uint8)
            masks.append(mask)

        # Combine masks with object IDs
        final_mask = np.zeros((self.embedding_size, self.embedding_size), dtype=np.uint8)
        for i, obj_mask in enumerate(masks, 1):
            final_mask = np.where(final_mask == 0, obj_mask * i, final_mask)
            
    
        if random.random()<0.9:
            return np.repeat(np.expand_dims(final_mask, axis=0), self.embedding_dim, axis=0)
        else:
            return np.zeros((self.embedding_dim, self.embedding_size, self.embedding_size), np.uint8)