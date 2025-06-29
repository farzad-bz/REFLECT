
# ✨ REFLECT ✨
**A PyTorch Implementation for Unsupervised Brain Anomaly Detection**

This repository hosts the official PyTorch implementation for our paper accepted in MICCAI2025:  
"REFLECT: Rectified Flows for Efficient Brain Anomaly Correction Transport.

---

## 🎨 Approach

![REFLECT Method](./assets/method.png)

---

## ⚙️ Setup

### 🛠️ Environment

Our experiments run on **Python 3.11**. Install all the required packages by executing:

```bash
pip3 install -r requirements.txt
```

### 📁 Datasets

Prepare your data as follows:

1. **Data Registration & Preprocessing:**  
   - Register with MNI_152_1mm.
   - Preprocess, normalize, pad and extract axial slices.

2. **Dataset Organization:**  
   - Ensure **training** and **validation** sets contain only normal, healthy data.
   - **Test** set should include abnormal slices.
   - Organize your files using this structure:

   ```
   ├── Data
       ├── train
       │   ├── brain_scan_{train_image_id}_slice_{slice_idx}_{modality}.png
       │   ├── brain_scan_{train_image_id}_slice_{slice_idx}_brainmask.png
       │   └── ...
       ├── val
       │   ├── brain_scan_{val_image_id}_slice_{slice_idx}_{modality}.png
       │   ├── brain_scan_{val_image_id}_slice_{slice_idx}_brainmask.png
       │   └── ...
       └── test
           ├── brain_scan_{test_image_id}_slice_{slice_idx}_{modality}.png
           ├── brain_scan_{test_image_id}_slice_{slice_idx}_brainmask.png
           ├── brain_scan_{test_image_id}_slice_{slice_idx}_segmentation.png
           └── ...
   ```

---

## 🔧 Pretrained Weights & VAE Fine-Tuning

### Pretrained VAE Models

To jumpstart your experiments, we provide pretrained weights adapted for 1-channel medical brain images. These models are available on [HuggingFace](https://huggingface.co/farzadbz/Medical-VAE).

### Train & Fine-Tune VAE

If you prefer to train your own VAE from scratch, please refer to the [LDM-VAE repository](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#training-autoencoder-models) for detailed instructions.

---

## 🚄 Training REFLECT

To train REFLECT, run the following command. This configuration leverages a UNet_L model with data augmentation and integrates the pretrained VAE:

```bash
torchrun train_REFLECT.py \
            --model UNet_L \
            --image-size 256 \
            --augmentation True \
            --vae kl_f8 \
            --modality 1 \
            --ckpt-every 0 
```

---

## 📸 Sample Results


![Sample Results](./assets/results.png)
