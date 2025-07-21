# -*- coding: utf-8 -*-
"""
Real‑time Speck‑2e DVS → ANN optical‑flow inference
--------------------------------------------------
* Pulls raw address‑events from a Speck‑2e dev‑kit.
* Rasterises to a 2‑channel (polarity) 128×128 frame.
* Runs an **ONNX** optical‑flow decoder (events + hidden → flow + new_hidden).
* Computes quadrant‑level divergence and prints instant/average latency.
* Optionally shows a GUI via *jetson‑stats* Visualizer on `tcp://0.0.0.0:40000`.

Tested with:
  • speck2edevkit firmware ≥2.2
  • samna ≥ 0.10, samnagui ≥ 0.10
  • onnxruntime ≥ 1.17 (CPU EP)

Edit the paths to your model/checkpoint as needed.
"""
from __future__ import annotations

from multiprocessing import Process
import threading
import time
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import onnxruntime as ort
import samna
import samnagui
from sinabs.backend.dynapcnn.chip_factory import ChipFactory

# ───────────────────────────────────────────────────────────────
#  Configuration
# ───────────────────────────────────────────────────────────────
MODEL_ONNX = Path("Optical_Flow_tinycmax/qs95vlk2_minGRU_depth1_ANN.onnx")
HIDDEN_SHAPE = (1, 32, 16, 16)          # (B,C,H,W) – must match ONNX output
RASTER_SHAPE = (2, 128, 128)            # (polarity,H,W)
DVS_DT_MS = 10                           # How often to pull events (ms)
VISUALIZER = True                        # Set False to disable GUI

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ───────────────────────────────────────────────────────────────
#  Divergence helpers (re‑use from Divergence Quadrant)
# ───────────────────────────────────────────────────────────────

def divergence_ratio(D_pix: float, D_tip: float, eps: float = 1e-6) -> float:
    return (D_tip - D_pix) / (D_pix + eps)


def are_vectors_similar(vectors: List[Tuple[float, float]], deg_thresh: float = 15.0,
                        min_aligned: int = 3, mag_eps: float = 1e-3) -> bool:
    angles: List[float] = []
    for u, v in vectors:
        mag = np.hypot(u, v)
        if mag < mag_eps:
            continue
        angles.append(np.degrees(np.arctan2(v, u)) % 360)
    if len(angles) < min_aligned:
        return False
    ang = np.asarray(angles)
    for base in ang:
        diff = np.abs(((ang - base + 180) % 360) - 180)
        if np.count_nonzero(diff < deg_thresh) >= min_aligned:
            return True
    return False


def quadrant_means(flow: torch.Tensor):
    _, H, W = flow.shape
    mid_x, mid_y = W // 2, H // 2
    quads = [
        (slice(0, mid_y), slice(0, mid_x)),        # TL
        (slice(0, mid_y), slice(mid_x, W)),        # TR
        (slice(mid_y, H), slice(mid_x, W)),        # BR
        (slice(mid_y, H), slice(0, mid_x)),        # BL
    ]
    centres = [
        (W // 4, H // 4),
        (3 * W // 4, H // 4),
        (3 * W // 4, 3 * H // 4),
        (W // 4, 3 * H // 4),
    ]
    vecs: List[Tuple[float, float]] = []
    for sl_y, sl_x in quads:
        u = flow[0, sl_y, sl_x].mean().item()
        v = flow[1, sl_y, sl_x].mean().item()
        vecs.append((u, v))
    return centres, vecs

# neighbour list (clockwise wrap‑around)
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]

# ───────────────────────────────────────────────────────────────
#  Inference thread
# ───────────────────────────────────────────────────────────────

divergence_history: List[float] = []
latency_history: List[float] = []


def inference_loop(sink: samna.graph.Sink):
    """Blocking loop: pull events → raster → ONNX → divergence."""
    print("[inference] thread started")
    ort_session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])

    hidden = torch.zeros(*HIDDEN_SHAPE, dtype=torch.float32)

    while True:
        # 1) fetch events (blocking up to DVS_DT_MS ms)
        events = sink.get_events_blocking(DVS_DT_MS)
        t0 = time.time()
        if events:
            print(events)
            raster = ChipFactory.events_to_raster_fast(events=events, shape=RASTER_SHAPE)
        else:
            raster = torch.zeros(RASTER_SHAPE, dtype=torch.float32)
        raster = raster.unsqueeze(0)                  # add batch

        # 2) ONNX forward
        ort_inputs = {
            "events": raster.numpy(),
            "hidden": hidden.numpy(),
        }
        flow_np, hidden_np = ort_session.run(None, ort_inputs)
        hidden = torch.from_numpy(hidden_np)
        flow = torch.from_numpy(flow_np.squeeze(0))   # (2,H,W)

        # 3) Divergence
        centres, vecs = quadrant_means(flow)
        if are_vectors_similar(vecs):
            div = 0.0
        else:
            ratios = []
            for i, j in PAIRS:
                x1, y1 = centres[i]
                x2, y2 = centres[j]
                u1, v1 = vecs[i]
                u2, v2 = vecs[j]
                d_pix = np.hypot(x1 - x2, y1 - y2)
                d_tip = np.hypot((x1+u1)-(x2+u2), (y1+v1)-(y2+v2))
                ratios.append(divergence_ratio(d_pix, d_tip))
            div = float(np.mean(ratios))
        divergence_history.append(div)

        # 4) latency stats
        t1 = time.time()
        latency = t1 - t0
        latency_history.append(latency)
        inst_rate = 1 / latency
        avg_lat = sum(latency_history) / len(latency_history)
        avg_rate = 1 / avg_lat

        print(f"[inference] Divergence {div:+.4f} | Inst {inst_rate:5.1f} Hz | Avg {avg_rate:5.1f} Hz")


# ───────────────────────────────────────────────────────────────
#  Speck‑2e setup & GUI
# ───────────────────────────────────────────────────────────────

def open_speck2e():
    return samna.device.open_device("Speck2eDevKit:0")


def open_visualizer(endpoint="tcp://0.0.0.0:40000", w=0.75, h=0.75):
    proc = Process(target=samnagui.run_visualizer, args=(endpoint, w, h))
    proc.start()
    return proc


def main():
    # 1) GUI (optional)
    endpoint = "tcp://0.0.0.0:40000"
    if VISUALIZER:
        gui_proc = open_visualizer(endpoint)
    else:
        gui_proc = None

    # 2) Open dev‑kit and sink node
    dk = open_speck2e()
    graph = samna.graph.EventFilterGraph()

    # DVS raw‑monitor enable
    cfg = samna.speck2e.configuration.SpeckConfiguration()
    cfg.dvs_layer.raw_monitor_enable = True
    dk.get_model().apply_configuration(cfg)

    sink = samna.graph.sink_from(dk.get_model_source_node())

    graph.start()

    # 3) Start inference thread
    thr = threading.Thread(target=inference_loop, args=(sink,), daemon=True)
    thr.start()

    # 4) Keep alive until GUI closed (or Ctrl‑C)
    try:
        while gui_proc is None or gui_proc.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[main] Ctrl‑C received – exiting…")
    finally:
        graph.stop()
        samna.device.close_device(dk)
        if gui_proc is not None:
            gui_proc.terminate(); gui_proc.join()
        print("[main] Clean shutdown – bye!")


if __name__ == "__main__":
    main()




# from multiprocessing import Process
# import sinabs.backend.dynapcnn as sindynapcnn
# import sinabs.from_torch
# import torch
# import torch.nn as nn
# import samna
# import samnagui
# from sinabs.from_torch import from_model
# import numpy as np
# import time, timeit
# import torch.nn.functional as F
# from sinabs.backend.dynapcnn.chip_factory import ChipFactory
# import matplotlib.pyplot as plt
# import os
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
# import cv2
# from matplotlib.colors import hsv_to_rgb
# import threading
# import onnxruntime as ort
# import pickle
# import csv
# import sinabs.backend.dynapcnn.io as sio

# checkpoint = torch.load("Optical_Flow_tinycmax/qs95vlk2_minGRU_depth1_model.pt", map_location="cpu")
# print("Checkpoint Keys:", checkpoint.keys())

# #---------------Making model ready--------------------------------------------

# # onnx model check 
# #onnx_session = ort.InferenceSession("Optical_Flow_tinycmax/mem_decoder_network.onnx", providers=["CPUExecutionProvider"])
# onnx_session = ort.InferenceSession("Optical_Flow_tinycmax/qs95vlk2_minGRU_depth1_ANN.onnx", providers=["CPUExecutionProvider"])

# #-------------------------------------------
# def divergence_ratio(D_pixels: float, D_tips: float, eps: float = 1e-6) -> float:
#     """Signed fractional change of the distance between two vectors’ origins."""
#     return (D_tips - D_pixels) / (D_pixels + eps)

# def are_vectors_similar(vectors, deg_threshold: float = 15.0,
#                         min_aligned: int = 3, mag_eps: float = 1e-3) -> bool:
#     """True if ≥ min_aligned vectors point within deg_threshold ° of one another."""
#     angles = []
#     for u, v in vectors:
#         mag = np.hypot(u, v)
#         if mag < mag_eps:
#             continue
#         angles.append(np.degrees(np.arctan2(v, u)) % 360)

#     if len(angles) < min_aligned:
#         return False

#     angles = np.asarray(angles)
#     for base in angles:
#         diff = np.abs(((angles - base + 180) % 360) - 180)
#         if np.count_nonzero(diff < deg_threshold) >= min_aligned:
#             return True
#     return False

# def quadrant_means(flow: torch.Tensor):
#     """Return quadrant centres & mean flow vectors (TL, TR, BR, BL)."""
#     _, H, W = flow.shape
#     mid_x, mid_y = W // 2, H // 2
#     quads = [
#         (slice(0, mid_y), slice(0, mid_x)),        # TL  (0)
#         (slice(0, mid_y), slice(mid_x, W)),        # TR  (1)
#         (slice(mid_y, H), slice(mid_x, W)),        # BR  (2)
#         (slice(mid_y, H), slice(0, mid_x)),        # BL  (3)
#     ]
#     centres = [
#         (W // 4, H // 4),
#         (3 * W // 4, H // 4),
#         (3 * W // 4, 3 * H // 4),
#         (W // 4, 3 * H // 4),
#     ]
#     vectors = []
#     for sl_y, sl_x in quads:
#         u_mean = flow[0, sl_y, sl_x].mean().item()
#         v_mean = flow[1, sl_y, sl_x].mean().item()
#         vectors.append((u_mean, v_mean))
#     return centres, vectors


# #--------------------------------------------




# output_hidden_list = [None]
# trajectory = []
# divergence_history = []          # will store one value per optical‑flow frame
# quadrant_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)] 
# latency_accumulator = []

# def inference():
#     global output_hidden_list
#     global trajectory
#     global divergence_history, quadrant_pairs


#     def create_gaussian_kernel(kernel_size=5, sigma=0.3):
#         """Creates a 2D Gaussian kernel for convolution."""
#         grid = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
#         kernel_1d = torch.exp(-0.5 * (grid / sigma) ** 2)
#         kernel_1d /= kernel_1d.sum()  # Normalize

#         # Create 2D Gaussian filter
#         kernel_2d = torch.outer(kernel_1d, kernel_1d)
#         kernel_2d /= kernel_2d.sum()  # Normalize
    
#         return kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

#     def gaussian_smooth(frames, kernel_size=5, sigma=0.3):
#         """Applies Gaussian smoothing to the input raster frames using convolution."""
#         kernel = create_gaussian_kernel(kernel_size, sigma).to(frames.device)

#         # Reshape frames to (B, C=1, H, W) for convolution
#         frames = frames.permute(1,0,2,3).float()  # (64, 1, 16, 16)

#         # Apply Gaussian blur using 2D convolution
#         smoothed_frames = F.conv2d(frames, kernel, padding=kernel_size // 2, groups=1)
    
#         return smoothed_frames.squeeze(1)
    
#     def flow_map_to_image(frame, new_size=None):

#         # check shape
#         assert frame.ndim == 3 and frame.shape[0] == 2, "Flow must have shape (2, height, width)."

#         # flow magnitude
#         mag = (frame**2).sum(0) ** 0.5
#         min_mag = mag.min()
#         d_mag = mag.max() - min_mag

#         # flow angle
#         x, y = frame[0], frame[1]
#         ang = np.arctan2(y, x) + np.pi
#         ang *= 1.0 / np.pi / 2.0

#         # flow color
#         frame_hsv = np.stack([ang, np.ones_like(ang), mag - min_mag], axis=2)
#         frame_hsv[:, :, 2] /= d_mag if d_mag != 0.0 else 1.0

#         # to rgb ints
#         frame_rgb = hsv_to_rgb(frame_hsv)
#         frame_rgb = (frame_rgb * 255).astype(np.uint8)

#         if new_size is not None:
#             frame_rgb = cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_LINEAR)

#         return frame_rgb
    
#     # drwaing quiver as well, lets see
#     def draw_quiver_on_flow_image(flow_map, flow_image, step=8, scale=1.0, color=(0, 255, 0)):
#         h, w = flow_map.shape[1:]

#         for y in range(0, h, step):
#             for x in range(0, w, step):
#                 fx, fy = flow_map[0, y, x], flow_map[1, y, x]
#                 end_x = int(x + fx * scale)
#                 end_y = int(y + fy * scale)
#                 cv2.arrowedLine(flow_image, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

#         return flow_image
    
    
#     while True:

#         data = sink.get_events_blocking(10)
#         print(len(data))

#         if len(data) > 0:
#             timestamps, events = zip(*data)
#             readout_time = min(timestamps)


#             raster = ChipFactory.events_to_raster_fast(events=events,shape=(2,128,128))   # self=ChipFactory, dt = 0.01,     
#             #raster = ((gaussian_smooth(raster)).unsqueeze(0))
 
#         else:

#             readout_time = time.time()
#             raster = torch.zeros(1, 2, 128, 128)

            
#         ####  Model Inference ##---------------------------------
#         output_hidden = output_hidden_list[-1]
#             #input_dict = {"events": raster}
            
#         # for onnx
#         events_np = raster.cpu().numpy()

#         if output_hidden is not None:
#                 hidden_np = output_hidden.cpu().numpy()
#         else:
#                 hidden_np = np.zeros_like(events_np)

#         onnx_inputs = {
#                     "events": events_np,
#                     "hidden": hidden_np
#         }

#         onnx_outputs = onnx_session.run(None, onnx_inputs)
#         flow_map_np = onnx_outputs[0].squeeze(0)
#         print(flow_map_np.shape)
#         new_hidden_np = onnx_outputs[1]

#         flow_map = torch.tensor(flow_map_np)
#         output_hidden = torch.tensor(new_hidden_np)

#         centres, vectors = quadrant_means(flow_map)

#         # 3‑out‑of‑4 means almost parallel → zero divergence
#         if are_vectors_similar(vectors, deg_threshold=15.0, min_aligned=3):
#             divergence_val = 0.0
#         else:
#             ratios = []
#             for i, j in quadrant_pairs:
#                 x1, y1 = centres[i]
#                 x2, y2 = centres[j]
#                 u1, v1 = vectors[i]
#                 u2, v2 = vectors[j]

#                 D_pixels = np.hypot(x1 - x2, y1 - y2)
#                 D_tips   = np.hypot((x1 + u1) - (x2 + u2),
#                                 (y1 + v1) - (y2 + v2))
#                 ratios.append(divergence_ratio(D_pixels, D_tips))
#             divergence_val = float(np.mean(ratios))

#         divergence_history.append(divergence_val)
#         print(f"[inference] Divergence: {divergence_val:+.4f}")

        
#         trajectory.append(flow_map_np)
       
         

#         output_hidden_list.append(output_hidden)

#         inference_end = time.time()

#         ### Total latency:
#         total_latency = inference_end - readout_time
#         latency_accumulator.append(total_latency)
#         inst_rate = 1 / total_latency
#         avg_latency = sum(latency_accumulator) / len(latency_accumulator)
#         avg_rate   = 1 / avg_latency
#         print(f"[inference] Instant: {inst_rate:6.2f} Hz   "
#             f"Running‑avg: {avg_rate:6.2f} Hz "
#             f"({avg_latency*1000:.1f} ms)")


              

# def open_speck2e():
#     return samna.device.open_device("Speck2eDevKit:0")

# def open_visualizer(streamer_endpoint, window_width=0.75, window_height=0.75):
#     gui_process = Process(
#         target=samnagui.run_visualizer,
#         args=(streamer_endpoint, window_width, window_height), #,
#     )
#     gui_process.start()

#     return gui_process

# def build_samna_event_route(graph, dk):
#     # build a graph in samna to show dvs
#     _, _, streamer = graph.sequential(
#         [dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"]
#     )

#     streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")
#     if streamer.wait_for_receiver_count() == 0:
#         raise Exception(f'connecting to visualizer on {"tcp://0.0.0.0:40000"} fails')

#     return streamer

# devkit_name = "speck2edevkit:0"
# streamer_endpoint = "tcp://0.0.0.0:40000"


# gui_process = open_visualizer(streamer_endpoint, 0.75, 0.75)



# dk = open_speck2e()


# graph = samna.graph.EventFilterGraph()
# devkit_config = samna.speck2e.configuration.SpeckConfiguration()
# devkit_config.dvs_layer.raw_monitor_enable = True
# dk.get_model().apply_configuration(devkit_config)
# sink = samna.graph.sink_from(dk.get_model_source_node())


# graph.start()

# # lets put here
# inference_thread = threading.Thread(target=inference, daemon=True)
# inference_thread.start()

# #gui_process.join()
# try:
#     # poll until either the GUI exits or stop_event is set
#     while gui_process.is_alive():
#         time.sleep(0.1)
# finally:
#     # clean up everything else                   # signal the inference thread to quit
#     inference_thread.join()       
#     # ps = event_sink.get_events() 
#     graph.stop()                           # stop the Samna graph
#     samna.device.close_device(dk)          # close the hardware
#     print("Cleanup complete — visualizer window is still open.")
#     gui_process.terminate()
#     gui_process.join()

