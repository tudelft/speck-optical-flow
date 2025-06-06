
import pandas as pd
import torch
import imageio
import numpy as np
import os
from PIL import Image

img_folder = "/home/manu/Desktop/SPECK/Optical_Flow_tinycmax/images"
output_video_path = "/home/manu/Desktop/SPECK/Optical_Flow_tinycmax/events_hand_vid_resize.mp4"



df = pd.read_csv("/home/manu/Desktop/SPECK/Optical_Flow_tinycmax/data/events_hand.csv")

time_window = 10000
t0 = df["t"][0]
tk = df["t"].iloc[-1]
t_start, t_end = t0, tk
n_full_windows = max(1, int((t_end - t_start) / time_window))
print(n_full_windows)
linspace = np.linspace(t_start, n_full_windows * time_window + t_start, n_full_windows + 1)

t_start, t_end = linspace[:-1], linspace[1:]
t_start, t_end = t_start[:1500], t_end[:1500]

all_dfs = [
    df[(df["t"] >= start) & (df["t"] < end)]
    for start, end in zip(t_start, t_end)
]

print(len(all_dfs))

infer_count = 0
#Iterating over whole dataframe
for df in all_dfs: 
        
        frame = torch.zeros((128, 128, 3), dtype=torch.uint8)
        
        for _, row in df.iterrows():

            frame[row["y"], row["x"], row["p"]] += 1

        frame = frame / (max(1, frame.max() - frame.min())) * 255    

        imageio.imwrite(f'{img_folder}/frame_{infer_count:05d}.png', frame.detach().cpu().numpy().astype(np.uint8))

        infer_count += 1  

        print(f'Created image: {img_folder}/{infer_count}.png')

image_files = sorted([img for img in os.listdir(img_folder) if img.endswith('.png')])

fps = 100

# # Create a video writer with imageio
# with imageio.get_writer(output_video_path, fps=fps) as writer:  
#         for image_file in image_files:
#             # Load image and append to video
#             image = imageio.imread(os.path.join(img_folder, image_file))
#             writer.append_data(image)

# print(f"Video saved as events_hand_vid") 

new_size = (600, 600) 

with imageio.get_writer(output_video_path, fps=fps) as writer:
    for image_file in image_files:
        # Load image using imageio
        image = imageio.imread(os.path.join(img_folder, image_file))
        
        # Resize image using PIL before appending it to the video
        pil_image = Image.fromarray(image)  # Convert the image to a PIL object
        pil_image_resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)  # Resize the image
        image_resized = np.array(pil_image_resized)  # Convert it back to a NumPy array
        
        # Append the resized image to the video
        writer.append_data(image_resized)

print(f"Video saved as {output_video_path}")

        