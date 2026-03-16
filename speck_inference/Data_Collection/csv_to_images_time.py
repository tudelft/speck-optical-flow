import pandas as pd
import torch
import imageio.v2 as imageio
import numpy as np
import os
from PIL import Image

csv_path    = "/home/manu-singh/Speck_Optical_Flow/data_optical_flow/events_check4.csv"
img_folder  = "/home/manu-singh/Speck_Optical_Flow/data_optical_flow/frames"
output_video_path = "/home/manu-singh/Speck_Optical_Flow/data_optical_flow/events_video4.mp4"

os.makedirs(img_folder, exist_ok=True)

df = pd.read_csv(csv_path)

time_window = 10000  # microseconds per frame
t0 = df["t"].iloc[0]
tk = df["t"].iloc[-1]
n_full_windows = max(1, int((tk - t0) / time_window))
print(f"Total time windows (frames): {n_full_windows}")

linspace = np.linspace(t0, n_full_windows * time_window + t0, n_full_windows + 1)
t_start, t_end = linspace[:-1], linspace[1:]
# MS: t_start, t_end = t_start[:1500], t_end[:1500]  — removed hardcoded frame cap

# Assign each event to a frame index
df["frame_idx"] = ((df["t"] - t0) / time_window).astype(int).clip(0, n_full_windows - 1)

print(f"Generating {n_full_windows} frames...")

for infer_count in range(n_full_windows):
    frame_df = df[df["frame_idx"] == infer_count]
    frame = np.zeros((128, 128, 3), dtype=np.float32)

    if not frame_df.empty:
        np.add.at(frame, (frame_df["y"].values, frame_df["x"].values, frame_df["p"].values), 1)

    frame_max = frame.max()
    if frame_max > 0:
        frame = frame / frame_max * 255

    imageio.imwrite(
        f'{img_folder}/frame_{infer_count:05d}.png',
        frame.astype(np.uint8)
    )

    if infer_count % 100 == 0:
        print(f'Created frame {infer_count}/{n_full_windows}')

# Compile frames into video
image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])

# Real-time fps: 1 frame = time_window microseconds, so fps = 1e6 / time_window
fps = 1_000_000 / time_window
print(f"Real-time playback fps: {fps}")

new_size = (608, 608)  # divisible by 16 for codec compatibility

with imageio.get_writer(output_video_path, fps=fps) as writer:
    for image_file in image_files:
        image = imageio.imread(os.path.join(img_folder, image_file))
        pil_image = Image.fromarray(image)
        pil_image_resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        writer.append_data(np.array(pil_image_resized))

print(f"Video saved to {output_video_path}")
