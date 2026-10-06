# -*- coding: utf-8 -*-
"""
Real‑time Speck‑2e DVS → ANN optical‑flow inference
--------------------------------------------------
Fix 2025‑07‑21:
    * **ChipFactory.events_to_raster_fast** expects a four‑dimensional
      output tensor indexed as  [0, fs, ys, xs].  Therefore the raster shape
      must be **(1, 2, 128, 128)** – *batch/time*, *polarity*, *H*, *W* – not
      the previous (2, 128, 128).
    * Removed the extra *unsqueeze(0)* after rasterisation.

Everything else (ONNX I/O, divergence math, visualiser) is unchanged.
"""
from __future__ import annotations
from pathlib import Path
RUN_DIR = Path(__file__).resolve().parent  # model weights (.pt/.onnx) live here
DATA_DIR = Path(__file__).resolve().parent / "data"  # recorded/generated arrays
DATA_DIR.mkdir(exist_ok=True)


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
MODEL_ONNX = Path(f"{RUN_DIR}/qs95vlk2_minGRU_depth1_ANN.onnx")
HIDDEN_SHAPE = (1, 32, 16, 16)          # (B,C,H,W) – must match ONNX output
RASTER_SHAPE = (1, 2, 128, 128)         # **fixed** (batch,polarity,H,W)
DVS_DT_MS = 10                           # How often to pull events (ms)
VISUALIZER = True                        # Set False to disable GUI

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ───────────────────────────────────────────────────────────────
#  Divergence helpers  (unchanged)
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
        (slice(0, mid_y), slice(0, mid_x)),
        (slice(0, mid_y), slice(mid_x, W)),
        (slice(mid_y, H), slice(mid_x, W)),
        (slice(mid_y, H), slice(0, mid_x)),
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

PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]

# ───────────────────────────────────────────────────────────────
#  Inference thread
# ───────────────────────────────────────────────────────────────

divergence_history: List[float] = []
latency_history: List[float] = []

event_count_history = []     # one integer per frame
frame_index_history = [] 


def events_to_raster_fast(events, shape: Tuple[int, int, int] = (2, 128, 128)) -> torch.Tensor:
    """Vectorised, *batch‑aware* rasteriser for Speck‑2e DVS events.

    Args
    ----
    events : iterable of samna Speck2eDvsEvent
        The raw polarity events.
    shape  : (C, H, W)
        Desired per‑batch raster shape (*without* the batch axis).  `C` must
        match the number of polarity channels you want (default 2).

    Returns
    -------
    torch.Tensor
        Tensor with shape **(1, C, H, W)** – leading dim reserved for future
        batching but always 1 here.
    """
    C, H, W = shape
    raster_np = np.zeros((C, H, W), dtype=np.float32)

    if events:
        p   = np.fromiter((e.p for e in events), dtype=np.int16)
        xs  = np.fromiter((e.x for e in events), dtype=np.int16)
        ys  = np.fromiter((e.y for e in events), dtype=np.int16)

        # Keep only events that fall inside our raster cube
        mask = (p >= 0) & (p < C) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if mask.any():
            np.add.at(raster_np, (p[mask], ys[mask], xs[mask]), 1.0)

    return torch.from_numpy(raster_np).unsqueeze(0)  # (1,C,H,W)


def inference_loop(sink: samna.graph.Sink):
    print("[inference] thread started")
    ort_session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
    hidden = torch.zeros(*HIDDEN_SHAPE, dtype=torch.float32)

    while True:
        t0 = time.time()
        events = sink.get_events_blocking(DVS_DT_MS)
        

        raster = events_to_raster_fast(events)   # (1,2,128,128)

        ort_inputs = {
            "events": raster.numpy().astype(np.float32),
            "hidden": hidden.numpy().astype(np.float32),
        }
        flow_np, hidden_np = ort_session.run(None, ort_inputs)
        hidden = torch.from_numpy(hidden_np)
        flow = torch.from_numpy(flow_np.squeeze(0))           # (2,H,W)

        # Divergence (unchanged)
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

        # latency
        latency = time.time() - t0             # seconds for this frame
        event_cnt = len(events)                # how many events in the batch

        frame_idx = len(frame_index_history)   # 0‑based index

        # push to history buffers
        latency_history.append(latency)
        event_count_history.append(event_cnt)
        frame_index_history.append(frame_idx)

        # --------- live FPS print -----------------------------------------
        inst_rate = 1.0 / latency
        avg_rate  = frame_idx / sum(latency_history) if frame_idx > 0 else inst_rate

        print(f"[inference] f#{frame_idx:5d} | "
            f"div {div:+.4f} | "
            f"inst {inst_rate:6.1f} Hz | "
            f"avg {avg_rate:6.1f} Hz | "
            f"events {event_cnt:6d}")



# ───────────────────────────────────────────────────────────────
#  Speck‑2e setup & GUI
# ───────────────────────────────────────────────────────────────

def open_speck2e():
    return samna.device.open_device("Speck2eDevKit:0")

def build_samna_event_route(graph, dk):
    # build a graph in samna to show dvs
    _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"]
    )

    streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {"tcp://0.0.0.0:40000"} fails')

    return streamer

def configure_visualizer(graph, streamer):
    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
    #graph.start()
    
    visualizer_config = samna.ui.VisualizerConfiguration(
        # add plots to gui
        plots=[
            # add plot to show pixels
            samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1.0, 1.0]),]
            )
    return config_source, visualizer_config


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
    streamer = build_samna_event_route(graph, dk)
    # DVS raw‑monitor enable
    cfg = samna.speck2e.configuration.SpeckConfiguration()
    cfg.dvs_layer.raw_monitor_enable = True
    dk.get_model().apply_configuration(cfg)

    model_source = dk.get_model_source_node()
    config_source, visualizer_config = configure_visualizer(graph, streamer)
    config_source.write([visualizer_config])

    sink = samna.graph.sink_from(model_source)
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
        
        import matplotlib.pyplot as plt

        if latency_history:                            # avoid empty plots
            fig, ax1 = plt.subplots()
            ax1.plot(frame_index_history,
                     [1.0/lt for lt in latency_history],
                    label="Instant FPS", linewidth=1)
            ax1.set_xlabel("Frame #")
            ax1.set_ylabel("FPS", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.plot(frame_index_history,
                    event_count_history,
                    label="# events", color="tab:red", linewidth=1, alpha=0.7)
            ax2.set_ylabel("Events per frame", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            fig.suptitle("Frame‑by‑frame performance")
            fig.tight_layout()
            plt.savefig("latency_vs_events_ANN.png")


if __name__ == "__main__":
    main()




