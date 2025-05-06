import numpy as np

def patchify_image(img, patch_size=128):
    """
    Splits an image into patches of size `patch_size x patch_size`,
    padding with zeros if needed.

    Args:
        img (np.ndarray): Input image as a 2D or 3D numpy array (H x W x C) or (H x W).
        patch_size (int): Size of the square patches (default: 128).

    Returns:
        List[np.ndarray]: List of image patches.
    """
    if img.ndim == 2:  # If grayscale, add a channel dimension
        img = img[:, :, np.newaxis]

    height, width, channels = img.shape

    # Calculate new padded size
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    # Pad the image with zeros
    padded_img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    patches = []
    new_height, new_width, _ = padded_img.shape

    for y in range(0, new_height, patch_size):
        for x in range(0, new_width, patch_size):
            patch = padded_img[y:y + patch_size, x:x + patch_size]
            patchito = patch[:, :, 0]
            patches.append(patch)

    return patches

def reconstruct_from_patches(patches_array, original_height=None, original_width=None, patch_size=128):
    """
    Reconstruct the image from 128x128 patches stored in a 3D array (128, 128, N_patches).

    Args:
        patches_array (np.ndarray): 3D array (128, 128, N_patches)
        original_height (int, optional): Original height before patching (to crop padding).
        original_width (int, optional): Original width before patching (to crop padding).
        patch_size (int): Patch size, default 128.

    Returns:
        np.ndarray: Reconstructed image.
    """
    patch_h, patch_w, num_patches = patches_array.shape

    # Find how many patches along width and height
    patches_per_row = int(np.ceil(np.sqrt(num_patches)))  # Assuming square-ish layout
    patches_per_col = int(np.ceil(num_patches / patches_per_row))

    # Create empty canvas
    full_height = patches_per_col * patch_size
    full_width = patches_per_row * patch_size

    reconstructed = np.zeros((full_height, full_width))

    patch_idx = 0
    for row in range(patches_per_col):
        for col in range(patches_per_row):
            if patch_idx >= num_patches:
                break
            y_start = row * patch_size
            x_start = col * patch_size
            reconstructed[y_start:y_start + patch_size, x_start:x_start + patch_size] = patches_array[:, :, patch_idx]
            patch_idx += 1

    # Crop back to original size if needed
    if original_height is not None and original_width is not None:
        reconstructed = reconstructed[:original_height, :original_width]

    # Parse the image back to integers
    reconstructed = reconstructed.astype(np.uint8)

    return reconstructed
