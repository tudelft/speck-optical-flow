from pathlib import Path
RUN_DIR = Path(__file__).resolve().parent.parent / "Optical_Flow_run"  # model weights (.pt/.onnx) live here
DATA_DIR = Path(__file__).resolve().parent / "data"  # recorded/generated arrays
DATA_DIR.mkdir(exist_ok=True)

import torch
import imageio
import pandas as pd
import os
import numpy as np

import os
import imageio

# img_folder = f"{DATA_DIR}/images/images_speck_4_4"
# output_video = f"{DATA_DIR}/images/images_speck_4_4/speck_4_4_vid.mp4"
# fps = 2

# # Function to extract numeric part for proper sorting
# def extract_index(filename):
#     return int(''.join(filter(str.isdigit, filename)))

# # Sort image files numerically
# image_files = sorted(
#     [img for img in os.listdir(img_folder) if img.endswith('.png')],
#     key=extract_index
# )

# # Create video
# with imageio.get_writer(output_video, fps=fps) as writer:
#     for image_file in image_files:
#         image = imageio.imread(os.path.join(img_folder, image_file))
#         writer.append_data(image)

# print(f"Video saved as {output_video}")


from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip

# Load the two video files
video1 = VideoFileClip(f"{DATA_DIR}/images/images_speck_4_3/speck_4_3_vid.mp4")
video2 = VideoFileClip(f"{DATA_DIR}/images/images_speck_4_4/speck_4_4_vid.mp4")
video3 = VideoFileClip(f"{DATA_DIR}/images/images_speck_no_rec/speck_no_rec_vid.mp4")
video4 = VideoFileClip(f"{DATA_DIR}/images/images_snn_4_3/snn_4_3_vid.mp4")
video5 = VideoFileClip(f"{DATA_DIR}/images/images_snn_4_4/snn_4_4_vid.mp4")
video6 = VideoFileClip(f"{DATA_DIR}/images/images_snn_no_rec/snn_no_rec_vid.mp4")



# Concatenate videos side by side
final_video = clips_array([[video1, video2, video3],
                           [video4, video5, video6]])

# final_video = clips_array([[video1, video2]])

# Save the output video
final_video.write_videofile(f"{DATA_DIR}/images/combined_vid.mp4", codec="libx264", fps=video1.fps)
