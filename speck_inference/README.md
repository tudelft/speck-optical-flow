# Speck inference and data collection

Code that runs on the Speck2e devkit side: event recording, ANN→SNN conversion of the trained encoder,
and live hybrid inference. The spiking encoder runs on Speck; the minGRU memory and decoder run on the host
(Jetson Orin NX in the paper) through ONNX Runtime. Install `sinabs` and `samna` first (`pip install -r requirements.txt`).

## Data collection

[`Data_Collection/data_collection_csv_final.py`](Data_Collection/data_collection_csv_final.py) records events
(x, y, t, p) from the Speck2e devkit to CSV in real time. Recordings are written to [`data_optical_flow/`](data_optical_flow).
[`Data_Collection/csv_to_images_time.py`](Data_Collection/csv_to_images_time.py) renders a recording into frames and a video.

## Live optical flow

All scripts are in [`Optical_Flow_run/`](Optical_Flow_run). They load weights from that folder and write any
arrays they save to `Optical_Flow_run/data/`, so they can be started from any working directory.

| Script | Model |
|--------|-------|
| `speck_inference_32ch_minGRU_divergence.py` | Hybrid Speck + ANN, minGRU 32 ch (model used in the paper) |
| `speck2f_inference_32ch_minGRU_divergence_withLatency.py` | Same, logging inference rate/latency |
| `ANN_inference_32ch_minGRU_divergence*.py` | ANN-only baseline for comparison |
| `speck_inference_64ch*.py`, `speck_inference_16ch_power.py` | Other channel widths / GRU variants evaluated in the paper |
| `onnx_model*.py` | Export the memory + decoder part of a checkpoint to ONNX |

The training code that produces these checkpoints is in [`../training`](../training).
The onboard ROS 2 node that wraps this pipeline for flight is in [`../speckflow_ros2`](../speckflow_ros2).

## Speck firmware

[`SINABS_STUFF/images_flash.py`](SINABS_STUFF/images_flash.py) re-flashes the devkit. The firmware images it expects
(`motherBoardV2_0_11_5.img`, `Speck2eDevKit_1_0_1_1_0.bin`) are not redistributed here; get them from SynSense.
[`SINABS_STUFF/60-synsense.rules`](SINABS_STUFF/60-synsense.rules) is the udev rule for USB access on Linux.
