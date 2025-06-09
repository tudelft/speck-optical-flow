
import torch
import torchvision.transforms as T
from PIL import Image
import os
import pickle


with open('Optical_Flow_tinycmax/data/hollow_rasters.npy', 'rb') as f:
    hollow_rasters = pickle.load(f)

# print(speck_output[10])

# for t, tensor in enumerate(speck_output):
#     tensor = tensor.squeeze(0)  # shape: [2, 4, 4]
    
#     # Normalize each channel independently to [0, 255]
#     red = tensor[0]
#     green = tensor[1]

#     def normalize_snn(img):
#         img = img - img.min()
#         if img.max() != 0:
#             img = img / img.max()
#         return (img * 255).byte()
    
#     def normalize_speck(img):
#         # Assumes img is a torch.Tensor with values like 0 and 1
#         return (img * 255).clamp(0, 255).byte()

#     red_img = normalize_speck(red)
#     green_img = normalize_speck(green)
#     blue_img = torch.zeros_like(red_img, dtype=torch.uint8)

#     # Stack into RGB [H, W, 3]
#     rgb_img = torch.stack([red_img, green_img, blue_img], dim=-1)  # shape: [4, 4, 3]

#     # Convert to PIL and upscale
#     pil_img = Image.fromarray(rgb_img.numpy(), mode='RGB')
#     pil_img = pil_img.resize((128, 128), Image.NEAREST)
        
#     pil_img.save(f"Optical_Flow_tinycmax/images/images_speck_4_4/t{t}.png")


os.makedirs("Optical_Flow_tinycmax/images/hollow_raster_images", exist_ok=True)

# Save rasters as grayscale images
for t, tensor in enumerate(hollow_rasters):
    # tensor shape: [1, 1, 16, 16] -> squeeze to [16, 16]
    img_tensor = tensor.squeeze(0).squeeze(0)  # shape: [16, 16]
    
    # Convert to [0, 255] uint8 image
    img = (img_tensor * 255).byte()  # Values are 0 or 1 → 0 or 255
    pil_img = Image.fromarray(img.numpy(), mode='L')
    
    # Optional: upscale to make image more visible
    pil_img = pil_img.resize((128, 128), Image.NEAREST)

    # Save the image
    pil_img.save(f"Optical_Flow_tinycmax/images/hollow_raster_images/step_{t:03d}.png")


