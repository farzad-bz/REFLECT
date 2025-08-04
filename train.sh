CUDA_VISIBLE_DEVICES=4 torchrun evaluate_REFLECT.py \
            --data-dir /home-2/ar94660/Datasets/Medical/Brats2023-slices/ \
            #model path
            --model-path /home-2/ar94660/REFLECT/REFLECT_BraTS_UNet_S_T1_256_kl_f8/000-UNet_S-T1/checkpoints/last.pt \