import os
import time
import json

import torch
from torchvision import transforms
from model.models_vit_tensor_CD import vit_base_patch16
import skimage.io as io
from OSCD_utils import patchify_image, reconstruct_from_patches

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from src import UNet
#
# from src import UNet
# import pydensecrf.densecrf as dcrf


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def open_image(img_path):
    # with rasterio.open(img_path) as data:
    #     img = data.read()  # (c, h, w)
    img = io.imread(img_path)

    # return img.transpose(1, 2, 0).astype(np.float32)
    return img.astype(np.float32)

def to_rgb(img):
    if img.ndim == 2:  # single band
        return np.stack([img]*3, axis=-1)
    elif img.shape[2] == 1:  # (H, W, 1)
        return np.repeat(img, 3, axis=2)
    elif img.shape[2] >= 3:  # more than 3 bands
        return img[..., 1:4]
    return img

def plot_patch_change_detection(img1, img2, mask, path=None):
    # Plot the grid
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(to_rgb(img1))
    axs[0].set_title("Before (img1)")
    axs[0].axis('off')

    axs[1].imshow(to_rgb(img2))
    axs[1].set_title("After (img2)")
    axs[1].axis('off')

    axs[2].imshow(mask)
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    data_path = os.path.join(os.getcwd(), "data/OSCD/Onera Satellite Change Detection dataset - Images")
    models_path = os.path.join("./", "models")
    palette_path = os.path.join("./", "downstream_predict", "OSCD", "palette.json")
    
    results_path = os.path.join("./", "results", "OSCD")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # assert os.path.exists(weights_path), f"weights {weights_path} not found."
    # assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = vit_base_patch16()
    # model = UNet(in_channels=12, num_classes=13, base_c=64)
    # model = UPerNet(num_classes=13)

    # delete weights about aux_classifier
    # weights_dict = torch.load(weights_path, map_location='cpu')['model']
    # load weights
    checkpoint = torch.load(os.path.join(models_path, "spectralGPT+_54_28_trained.pth"),
                            map_location=device)

    checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    # model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # load image
    image_folder_path1 = os.path.join(data_path, "abudhabi/imgs_1_rect")
    image_folder_path2 = os.path.join(data_path, "abudhabi/imgs_2_rect")

    # Get all image filenames from the image folder
    image_file_names = os.listdir(image_folder_path1)
    # Remove band B10 -> Stablished in section 2.8 [Implementation Details and Experimental Setup] of the paper
    # Filter out the B10.tif file from the image filenames
    image_file_names = [f for f in image_file_names if f != "B10.tif"]


    # Iterate over the image filenames
    for i, image_file_name in enumerate(image_file_names):

        img1 = open_image(os.path.join(image_folder_path1, image_file_name))
        img2 = open_image(os.path.join(image_folder_path2, image_file_name))

        # Save the images on results folder
        # Image.fromarray(img1).save(os.path.join("./results/OSCD/abudhabi", "before_" + image_file_name))
        # Image.fromarray(img2).save(os.path.join("./results/OSCD/abudhabi", "after_" + image_file_name))
        
        if i==0: # Initialize bands object
            img1_bands = np.expand_dims(img1, axis=2)
            img2_bands = np.expand_dims(img2, axis=2)
        else:
            img1_bands = np.concatenate((img1_bands, np.expand_dims(img1, axis=2)), axis=2)
            img2_bands = np.concatenate((img2_bands, np.expand_dims(img2, axis=2)), axis=2)

    kid1 = (img1_bands - img1_bands.min(axis=(0, 1), keepdims=True))
    mom1 = (img1_bands.max(axis=(0, 1), keepdims=True) - img1_bands.min(axis=(0, 1), keepdims=True))
    img1_bands = kid1 / (mom1)

    kid2 = (img2_bands - img2_bands.min(axis=(0, 1), keepdims=True))
    mom2 = (img2_bands.max(axis=(0, 1), keepdims=True) - img2_bands.min(axis=(0, 1), keepdims=True))
    img2_bands = kid2 / (mom2)

    # We patch the image to fit 128x128x12
    img1_patches = patchify_image(img1_bands)
    img2_patches = patchify_image(img2_bands)

    for i, (patch_1, patch_2) in enumerate(zip(img1_patches, img2_patches)):

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        patch_1_tensor = data_transform(patch_1)
        patch_1_tensor = torch.unsqueeze(patch_1_tensor, dim=0)
        patch_2_tensor = data_transform(patch_2)
        patch_2_tensor = torch.unsqueeze(patch_2_tensor, dim=0)

        patch_1_tensor = patch_1_tensor.cuda()
        patch_2_tensor = patch_2_tensor.cuda()

        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
                t_start = time_synchronized()
                patch_1_tensor_kk = patch_1_tensor[:, 0, :, :].unsqueeze(1)
                patch_2_tensor_kk = patch_2_tensor[:, 0, :, :].unsqueeze(1)
                output = model(patch_1_tensor.to(device),patch_2_tensor.to(device))
                t_end = time_synchronized()
                print("inference time: {}".format(t_end - t_start))

                prediction = output.argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                # print(prediction)
                if i == 0:
                    prediction_full = np.expand_dims(prediction, axis=2)
                else:
                    prediction_full = np.concatenate((prediction_full, np.expand_dims(prediction, axis=2)), axis=2)
        
        if np.sum(prediction == 1) > 1:
            plot_patch_change_detection(img1=patch_1, img2=patch_2, mask=prediction, path=os.path.join("./results/OSCD/abudhabi", f"patch_{i}.png"))

    # Reconstruct the full prediction image from patches
    prediction_full = reconstruct_from_patches(prediction_full, original_height=img1_bands.shape[0], original_width=img1_bands.shape[1])

    mask = Image.fromarray(prediction_full)
    mask.putpalette(pallette)
    mask.save(os.path.join("./results/OSCD/abudhabi", 'mask.tif'))

if __name__ == '__main__':
    main()
