from multiprocessing import Process
import sinabs.backend.dynapcnn as sindynapcnn
import sinabs.from_torch
import torch
import torch.nn as nn
import samna
import samnagui
from sinabs.from_torch import from_model
import numpy as np
import time, timeit
from sinabs.layers import IAFSqueeze
import torch.nn.functional as F
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2
from matplotlib.colors import hsv_to_rgb
from sinabs.utils import normalize_weights
import queue
import threading
import multiprocessing
import onnxruntime as ort
import pickle
import csv


checkpoint = torch.load("Optical_Flow_run/qs95vlk2_minGRU_depth1_model.pt", map_location="cpu")
print("Checkpoint Keys:", checkpoint.keys())

#---------------Making model ready--------------------------------------------

# onnx model check 
#onnx_session = ort.InferenceSession("Optical_Flow_tinycmax/mem_decoder_network.onnx", providers=["CPUExecutionProvider"])
onnx_session = ort.InferenceSession("Optical_Flow_run/qs95vlk2_minGRU_depth1.onnx", providers=["CPUExecutionProvider"])
#------------------------------------------------------------------

ann = nn.Sequential(
    nn.Conv2d(2, 8, kernel_size=7, padding=3, stride=2, bias=False),
    nn.ReLU(),
 
    nn.Conv2d(8, 16, kernel_size=3, padding=1,stride=2, bias=False),
    nn.ReLU(),

    nn.Conv2d(16, 32, kernel_size=3, padding=1,stride=2, bias=False),
    nn.ReLU(),

 # output is 32, 16, 16
)

ann[0].weight.data = checkpoint["network.encoder.enc1.synapse.weight"]
ann[2].weight.data = checkpoint["network.encoder.enc2.conv0.synapse.weight"]
ann[4].weight.data = checkpoint["network.encoder.enc2.conv1.synapse.weight"]

print(ann)

#---------------------------------------
ann = ann.to("cpu")

sinabs_model = sinabs.from_model(ann, add_spiking_output=True, batch_size=1).spiking_model.cpu()

### Scaling only in first layer. 
with torch.no_grad():
    for name, param in sinabs_model.named_parameters():
        if name == '0.weight':  
            param.data *= 1
            param_data0 = param.data

        if name == '2.weight':
            param.data *= 1
            param_data2 = param.data

        if name == '4.weight':
            param.data *= 1
            param_data4 = param.data

print(np.amax(np.array(param_data0)))
print(np.amax(np.array(param_data2)))
print(np.amax(np.array(param_data4)))

input_shape = (2, 128, 128)
dynapcnn_net = sindynapcnn.DynapcnnNetwork(
        sinabs_model.cpu(),
        input_shape=input_shape,
        discretize=True,
        dvs_input=True,)

#-------------------------------------------
def divergence_ratio(D_pixels: float, D_tips: float, eps: float = 1e-6) -> float:
    """Signed fractional change of the distance between two vectors’ origins."""
    return (D_tips - D_pixels) / (D_pixels + eps)

def are_vectors_similar(vectors, deg_threshold: float = 15.0,
                        min_aligned: int = 3, mag_eps: float = 1e-3) -> bool:
    """True if ≥ min_aligned vectors point within deg_threshold ° of one another."""
    angles = []
    for u, v in vectors:
        mag = np.hypot(u, v)
        if mag < mag_eps:
            continue
        angles.append(np.degrees(np.arctan2(v, u)) % 360)

    if len(angles) < min_aligned:
        return False

    angles = np.asarray(angles)
    for base in angles:
        diff = np.abs(((angles - base + 180) % 360) - 180)
        if np.count_nonzero(diff < deg_threshold) >= min_aligned:
            return True
    return False

def quadrant_means(flow: torch.Tensor):
    """Return quadrant centres & mean flow vectors (TL, TR, BR, BL)."""
    _, H, W = flow.shape
    mid_x, mid_y = W // 2, H // 2
    quads = [
        (slice(0, mid_y), slice(0, mid_x)),        # TL  (0)
        (slice(0, mid_y), slice(mid_x, W)),        # TR  (1)
        (slice(mid_y, H), slice(mid_x, W)),        # BR  (2)
        (slice(mid_y, H), slice(0, mid_x)),        # BL  (3)
    ]
    centres = [
        (W // 4, H // 4),
        (3 * W // 4, H // 4),
        (3 * W // 4, 3 * H // 4),
        (W // 4, 3 * H // 4),
    ]
    vectors = []
    for sl_y, sl_x in quads:
        u_mean = flow[0, sl_y, sl_x].mean().item()
        v_mean = flow[1, sl_y, sl_x].mean().item()
        vectors.append((u_mean, v_mean))
    return centres, vectors


#--------------------------------------------

collection0 = []
collection1 = []
collection2 = []

collection1_timestamps = []

last_readout_time = None

def custom_readout(collection):
    global last_readout_time
    global collection0, collection1

    now = time.time()
    collection_with_timestamp = {
        "timestamp": now,
        "events": collection
    }

    collection1.append(collection_with_timestamp)
    collection0.append(collection)

    now = time.time()
    if last_readout_time is not None:
        interval = now - last_readout_time
        frequency_hz = 1.0 / interval
        print(f"[custom_readout] Receiving collections at ~{frequency_hz:.2f} Hz")
    else:
        print("[custom_readout] First readout received")

    last_readout_time = now

    def generate_result(feature):
        e = samna.ui.Readout()
        e.feature = feature
        return [e]   
    return generate_result(int(0))

def get_latest_collection1():
    """Returns the full content of collection1 and clears it."""
    global collection1 
    if not collection1:
        return []
    

    # Flatten all appended batches
    combined = []
    for item in collection1:
        timestamp = item["timestamp"]
        events = item["events"]
        for event in events:
            combined.append((timestamp, event))
    
    collection1 = []  # clear after reading

    return combined


output_hidden_list = [None]
trajectory = []
divergence_history = []          # will store one value per optical‑flow frame
quadrant_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)] 
latency_accumulator = []
latency_history      = []   # seconds per frame
avg_rate_history     = []   # running‑avg Hz
event_count_history  = [] 
stop_event = threading.Event()

def inference(stop):
    global output_hidden_list
    global trajectory
    global divergence_history, quadrant_pairs
    global latency_history, avg_rate_history, event_count_history


    def create_gaussian_kernel(kernel_size=5, sigma=0.3):
        """Creates a 2D Gaussian kernel for convolution."""
        grid = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (grid / sigma) ** 2)
        kernel_1d /= kernel_1d.sum()  # Normalize

        # Create 2D Gaussian filter
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()  # Normalize
    
        return kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    def gaussian_smooth(frames, kernel_size=5, sigma=0.3):
        """Applies Gaussian smoothing to the input raster frames using convolution."""
        kernel = create_gaussian_kernel(kernel_size, sigma).to(frames.device)

        # Reshape frames to (B, C=1, H, W) for convolution
        frames = frames.permute(1,0,2,3).float()  # (64, 1, 16, 16)

        # Apply Gaussian blur using 2D convolution
        smoothed_frames = F.conv2d(frames, kernel, padding=kernel_size // 2, groups=1)
    
        return smoothed_frames.squeeze(1)
    
    def flow_map_to_image(frame, new_size=None):

        # check shape
        assert frame.ndim == 3 and frame.shape[0] == 2, "Flow must have shape (2, height, width)."

        # flow magnitude
        mag = (frame**2).sum(0) ** 0.5
        min_mag = mag.min()
        d_mag = mag.max() - min_mag

        # flow angle
        x, y = frame[0], frame[1]
        ang = np.arctan2(y, x) + np.pi
        ang *= 1.0 / np.pi / 2.0

        # flow color
        frame_hsv = np.stack([ang, np.ones_like(ang), mag - min_mag], axis=2)
        frame_hsv[:, :, 2] /= d_mag if d_mag != 0.0 else 1.0

        # to rgb ints
        frame_rgb = hsv_to_rgb(frame_hsv)
        frame_rgb = (frame_rgb * 255).astype(np.uint8)

        if new_size is not None:
            frame_rgb = cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_LINEAR)

        return frame_rgb
    
    # drwaing quiver as well, lets see
    def draw_quiver_on_flow_image(flow_map, flow_image, step=8, scale=1.0, color=(0, 255, 0)):
        h, w = flow_map.shape[1:]

        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow_map[0, y, x], flow_map[1, y, x]
                end_x = int(x + fx * scale)
                end_y = int(y + fy * scale)
                cv2.arrowedLine(flow_image, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

        return flow_image
    
    
    while not stop.is_set():

        data = get_latest_collection1()
        #print(len(data))

        if len(data) > 0:
            timestamps, events = zip(*data)
            readout_time = min(timestamps)


            raster = ChipFactory.events_to_raster_fast(events=events,shape=(32,16,16))   # self=ChipFactory, dt = 0.01,     
            #raster = ((gaussian_smooth(raster)).unsqueeze(0))
 
        # else:

        #     readout_time = time.time()
        #     raster = torch.zeros(1, 32, 16, 16)

            
            ####  Model Inference ##---------------------------------
            output_hidden = output_hidden_list[-1]
            #input_dict = {"events": raster}
            
            # for onnx
            events_np = raster.cpu().numpy()

            if output_hidden is not None:
                hidden_np = output_hidden.cpu().numpy()
            else:
                hidden_np = np.zeros_like(events_np)

            onnx_inputs = {
                    "events": events_np,
                    "hidden": hidden_np
            }

            onnx_outputs = onnx_session.run(None, onnx_inputs)
            flow_map_np = onnx_outputs[0].squeeze(0)
            #print(flow_map_np.shape)
            new_hidden_np = onnx_outputs[1]

            flow_map = torch.tensor(flow_map_np)
            output_hidden = torch.tensor(new_hidden_np)

            centres, vectors = quadrant_means(flow_map)

            # 3‑out‑of‑4 means almost parallel → zero divergence
            if are_vectors_similar(vectors, deg_threshold=15.0, min_aligned=3):
                divergence_val = 0.0
            else:
                ratios = []
                for i, j in quadrant_pairs:
                    x1, y1 = centres[i]
                    x2, y2 = centres[j]
                    u1, v1 = vectors[i]
                    u2, v2 = vectors[j]

                    D_pixels = np.hypot(x1 - x2, y1 - y2)
                    D_tips   = np.hypot((x1 + u1) - (x2 + u2),
                                (y1 + v1) - (y2 + v2))
                    ratios.append(divergence_ratio(D_pixels, D_tips))
                divergence_val = float(np.mean(ratios))

            divergence_history.append(divergence_val)
            print(f"[inference] Divergence: {divergence_val:+.4f}")

        
            trajectory.append(flow_map_np)
            # print(div.mean(), u, v)
        

            new_size = (600, 600)

            flow_image = flow_map_to_image(flow_map_np, new_size=new_size)#flow_image

            flow_map_resized = cv2.resize(flow_map_np.transpose(1, 2, 0), (600, 600))  # shape: (H, W, 2)
            flow_map_resized = flow_map_resized.transpose(2, 0, 1)  # shape: (2, H, W)

            # Draw arrows
            flow_with_quiver = draw_quiver_on_flow_image(flow_map_resized, flow_image.copy(), step=20, scale=5)
        

            # # Show image
            cv2.imshow("Optical Flow", flow_image)
            cv2.imshow("Optical Flow", flow_with_quiver)
            cv2.putText(flow_with_quiver,
                f"Divergence: {divergence_val:+.4f}",
                (10, 25),         # x, y
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.waitKey(1) 
         

            output_hidden_list.append(output_hidden)

            inference_end = time.time()

            ### Total latency:
            latency = inference_end - readout_time
            event_count = len(events)
        
            latency_history.append(latency)
            event_count_history.append(event_count)
            avg_hz = len(latency_history) / sum(latency_history)
            avg_rate_history.append(avg_hz)
        
            print(f"[inference] Div {divergence_val:+.4f} | "
                f"Avg {avg_hz:6.1f} Hz | "
                f"events {event_count:6d}")


              

feature_count = 32
def configure_visualizer(graph, streamer):
    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
    #graph.start()
    
    visualizer_config = samna.ui.VisualizerConfiguration(
        # add plots to gui
        plots=[
            # add plot to show pixels
            samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1.0, 1.0]),
            samna.ui.PowerMeasurementPlotConfiguration(
                    title="Power Consumption",
                    channel_count=5,
                    line_names=["io", "ram", "logic", "vddd", "vdda"],
                    layout=[0, 0.8, 1, 1],
                    show_x_span=10,
                    label_interval=2,
                    max_y_rate=1.5,
                    show_point_circle=False,
                    default_y_max=1,
                    y_label_name="power (mW)",
                )])
    return config_source, visualizer_config

def open_speck2f():
    return samna.device.open_device("Speck2fDevKit:0")

def open_visualizer(streamer_endpoint, window_width=0.75, window_height=0.75):
    gui_process = Process(
        target=samnagui.run_visualizer,
        args=(streamer_endpoint, window_width, window_height), #,
    )
    gui_process.start()

    return gui_process

def build_samna_event_route(graph, dk):
    # build a graph in samna to show dvs
    _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
    )

    streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {"tcp://0.0.0.0:40000"} fails')

    return streamer

devkit_name = "speck2fdevkit:0"
streamer_endpoint = "tcp://0.0.0.0:40000"


gui_process = open_visualizer(streamer_endpoint, 0.75, 0.75)
dynapcnn_net.to(device=devkit_name,monitor_layers=["dvs", -1],  # Last layer
                    chip_layers_ordering="auto",)

# here directly taking samna_config and later putting applying configuration
config = dynapcnn_net.samna_config
# MS: dynapcnn_net.chip_layers_ordering[-1] — chip_layers_ordering is a dict in newer sinabs, not a list
lyrs = list(dynapcnn_net.chip_layers_ordering.values())[-1]
# print(lyrs)
config.dvs_layer.monitor_enable = True
config.cnn_layers[lyrs].monitor_enable = True
config.dvs_layer.mirror.y = True  # correct lens natural y-inversion (up/down was flipped)


dk = open_speck2f()
dk.get_model().apply_configuration(config)


graph = samna.graph.EventFilterGraph()
streamer = build_samna_event_route(graph, dk)

(source ,readout_spike, spike_collection_filter, spike_count_filter,_) = graph.sequential(
                [
                    dk.get_model_source_node(),
                    "Speck2fOutputMemberSelect",
                    "Speck2fSpikeCollectionNode",
                    "Speck2fSpikeCountNode",
                    streamer,
                ]
        )

spike_collection_filter.set_interval_milli_sec(10)
readout_spike.set_white_list([lyrs], "layer")
spike_count_filter.set_feature_count(feature_count)

_, readout_filter, _ = graph.sequential([spike_collection_filter, "Speck2fCustomFilterNode", streamer])
readout_filter.set_filter_function(custom_readout)

#for getting output
# event_sink = samna.graph.sink_from(source)

## power details
power = dk.get_power_monitor()
power.start_auto_power_measurement(20)
power_source, _, _ = graph.sequential([power.get_source_node(), "MeasurementToVizConverter", streamer])
power_sink = samna.graph.sink_from(power_source)

def get_events():
        return power_sink.get_events()

# Configure the visualizer
config_source, visualizer_config = configure_visualizer(graph, streamer)
config_source.write([visualizer_config])

graph.start()

# lets put here
inference_thread = threading.Thread(target=inference, args=(stop_event,))
inference_thread.start()

#gui_process.join()
try:
    # poll until either the GUI exits or stop_event is set
    while gui_process.is_alive():
        time.sleep(0.1)
finally:
    # clean up everything else                   # signal the inference thread to quit
    stop_event.set()
    inference_thread.join()       
    readout_filter.stop()                  # stop the Samna filters
    # ps = event_sink.get_events() 
    graph.stop()                           # stop the Samna graph
    samna.device.close_device(dk)          # close the hardware
    print("Cleanup complete — visualizer window is still open.")
    gui_process.terminate()
    gui_process.join()

    import matplotlib.pyplot as plt

    # if avg_rate_history:                         # avoid empty plot
    #     fig, ax1 = plt.subplots()
    #     ax1.plot(avg_rate_history,
    #             color="tab:blue", label="avg Hz", linewidth=1)
    #     ax1.set_xlabel("Frame index")
    #     ax1.set_ylabel("Average Hz", color="tab:blue")
    #     ax1.tick_params(axis="y", labelcolor="tab:blue")

    #     ax2 = ax1.twinx()
    #     ax2.plot(event_count_history,
    #             color="tab:red", label="#events", linewidth=1, alpha=0.6)
    #     ax2.set_ylabel("Events per frame", color="tab:red")
    #     ax2.tick_params(axis="y", labelcolor="tab:red")

    #     fig.suptitle("DynapCNN pipeline – rate vs events")
    #     fig.tight_layout()
    #     plt.savefig("dynapcnn_rate.png")
# with open('Optical_Flow_tinycmax/data/flow_check/trajectory32_divergence14_7_2.npy', 'wb') as f:
#     pickle.dump(trajectory, f)

# csv_file_path_in_sink = 'Optical_Flow_tinycmax/data/input_events_div_14_7_2.csv'


# with open(csv_file_path_in_sink, mode='a', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         if csv_file.tell() == 0:
#             print("Writing header once...")
#             csv_writer.writerow(['layer','x', 'y', 't', 'p'])

# with open(csv_file_path_in_sink, mode='a', newline='') as csv_file:
#             csv_writer = csv.writer(csv_file)

#             # Process events
#             for event in ps:

#                 csv_writer.writerow([int(event.layer), int(event.x), int(event.y), int(event.timestamp), int(event.feature)])

# print("input events done!")

# ps = get_events()



# channel_data = [[] for _ in range(5)]
# for record in ps:
#     #print(record)
#     channel_data[record.channel].append((record.timestamp, record.value))

# import pickle

# with open('Optical_Flow_tinycmax/data/flow_maps_for_trajectory32.npy', 'wb') as f:
#     pickle.dump(trajectory, f)

# output_path = "Optical_Flow_tinycmax/data/power_data32.npy"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)

# numpy_arrays = [np.array(data) for data in channel_data]
# np.save(output_path, numpy_arrays)
