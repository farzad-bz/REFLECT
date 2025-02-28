from ldm.modules.diffusionmodules.openaimodel_medical import UNetModel


def create_unet_model(latent_size=32, model_channels=256, num_res_blocks=2, num_heads=8, channel_mult=[1,2,4], context_dim=None, in_channels=4, out_channels=4):
    model = UNetModel(image_size=latent_size, 
                    in_channels=in_channels,
                    model_channels=model_channels, 
                    out_channels=out_channels, 
                    num_heads=num_heads, 
                    num_res_blocks=num_res_blocks, 
                    dropout=0.4,
                    attention_resolutions=[4,2,1], 
                    channel_mult = channel_mult,
                    # num_head_channels= 32,
                    use_spatial_transformer= False,
                    transformer_depth= 1,
                    context_dim=context_dim)
    return model


def UNET_XS(in_channels=4, out_channels=4, **kwargs):
    return  create_unet_model(latent_size=32, model_channels=64, num_heads=4, channel_mult=[1,2,4], in_channels=in_channels, out_channels=out_channels)


def UNET_S(in_channels=4, out_channels=4, **kwargs):
    return  create_unet_model(latent_size=32, model_channels=128, num_heads=4, channel_mult=[1,2,4], in_channels=in_channels, out_channels=out_channels)


def UNET_M(in_channels=4, out_channels=4, **kwargs):
    return  create_unet_model(latent_size=32, model_channels=192, num_heads=6, channel_mult=[1,2,4], in_channels=in_channels, out_channels=out_channels)


def UNET_L(in_channels=4, out_channels=4, **kwargs):
    return  create_unet_model(latent_size=32, model_channels=256, num_heads=8, channel_mult=[1,2,4], in_channels=in_channels, out_channels=out_channels)


def UNET_XL(in_channels=4, out_channels=4, **kwargs):
    return  create_unet_model(latent_size=32, model_channels=384, num_heads=12, channel_mult=[1,2,4], in_channels=in_channels, out_channels=out_channels)




UNET_models = {
'UNet_XS' : UNET_XS, 
'UNet_S' : UNET_S, 
'UNet_M' : UNET_M, 
'UNet_L' : UNET_L, 
'UNet_XL' : UNET_XL, 
}

