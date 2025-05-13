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


checkpoint = torch.load("Optical_Flow_tinycmax/iterative_model_hand_16ch.pt", map_location="cpu")
print("Checkpoint Keys:", checkpoint.keys())

total_params = sum(p.numel() for p in checkpoint.values())
print(f"Total parameters: {total_params}")

#---------------Making model ready--------------------------------------------
import network_woenc_GRU as network_woenc_GRU

model = network_woenc_GRU.WrappedFlowNetwork(
    memory_channels=16,
    decoder_channels=16,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="reflect",
    scaling=32
)

state_dict = torch.load("Optical_Flow_tinycmax/iterative_model_hand_16ch.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
print(state_dict.keys())

## Skipping the encoder states
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("network.", "")  # strip the prefix
    if new_key.startswith("encoder."):
        continue 
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#------------------------------------------------------------------

ann = nn.Sequential(
    nn.Conv2d(2, 4, kernel_size=7, padding=3, stride=2, bias=False),
    nn.ReLU(),
 
    nn.Conv2d(4, 8, kernel_size=3, padding=1,stride=2, bias=False),
    nn.ReLU(),

    nn.Conv2d(8, 16, kernel_size=3, padding=1,stride=2, bias=False),
    nn.ReLU(),

 # output is 16, 16, 16
)

ann[0].weight.data = checkpoint["network.encoder.enc1.synapse.weight"]
ann[2].weight.data = checkpoint["network.encoder.enc2.conv0.synapse.weight"]
ann[4].weight.data = checkpoint["network.encoder.enc2.conv1.synapse.weight"]


print(ann)

# sample_data = torch.randn(1,2,128,128)
# output_layers = ['1','3', '5']  # Typically, these will be the ReLU layers
# param_layers = ['0', '2', '4']   # The preceding convolutional and fully connected layers to rescale

# normalize_weights(
#     ann=ann,               # Your trained CNN model
#     sample_data=sample_data,  # A sample batch of input data
#     output_layers=output_layers,  # List of layers to observe
#     param_layers=param_layers,    # List of layers whose parameters you want to adjust
#     percentile=99               # Percentile for normalization
# )


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

input_shape = (2, 128, 128)
dynapcnn_net = sindynapcnn.DynapcnnNetwork(
        sinabs_model.cpu(),
        input_shape=input_shape,
        discretize=True,
        dvs_input=True,)

#-------------------------------------------
collection0 = []
collection1 = []
collection2 = []

def custom_readout(collection):
    global collection1
    collection1.append(collection)

    global collection0

    collection0.append(collection)

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
        combined.extend(item)
    
    collection1 = []  # clear after reading
    return combined

output_hidden_list = [None]

def inference():
    global output_hidden_list
    global model
    global collection2

    def create_gaussian_kernel(kernel_size=5, sigma=0.3):
        """Creates a 2D Gaussian kernel for convolution."""
        grid = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (grid / sigma) ** 2)
        kernel_1d /= kernel_1d.sum()  # Normalize

        # Create 2D Gaussian filter
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()  # Normalize
    
        return kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    def gaussian_smooth(frames, kernel_size=3, sigma=0.3):
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
    
    while True:
        start_time = time.time()
        data = get_latest_collection1()
        print(len(data))
        end_time = time.time()
        duration = end_time - start_time
        rate = 1 / duration
        print(f"Spike rate: {rate:.2f} Hz") 

        if len(data) > 0:
            raster = ChipFactory.events_to_raster(self=ChipFactory, events=data, dt = 0.01,
                                          shape=(16,16,16))       
            raster = ((gaussian_smooth(raster)).unsqueeze(0))

            start_time = time.time()

            ####  Model Inference ##---------------------------------
            output_hidden = output_hidden_list[-1]
            input_dict = {"events": raster}

            with torch.no_grad():
                output_dict, output_hidden = model(input_dict, output_hidden)
                flow_map = output_dict["flow"].cpu()      
            flow_map_np = flow_map.cpu().numpy().squeeze() 
            # collection2.append(flow_map.cpu())

            new_size = (600, 600)

            flow_image = flow_map_to_image(flow_map_np, new_size=new_size)#flow_image

            flow_map_resized = cv2.resize(flow_map_np.transpose(1, 2, 0), (600, 600))  # shape: (H, W, 2)
            flow_map_resized = flow_map_resized.transpose(2, 0, 1)  # shape: (2, H, W)

            # Draw arrows
            flow_with_quiver = draw_quiver_on_flow_image(flow_map_resized, flow_image.copy(), step=20, scale=5)
        

            # # # Show image
            cv2.imshow("Optical Flow", flow_with_quiver)
            cv2.waitKey(1) 

            output_hidden_list.append(output_hidden)

            
            # rate = 1.0/ duration
            print(f"Spike rate: {duration:.2f} seconds")
              

feature_count = 16
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
                )]) # [0, 0, 1.0, 1.0])

    return config_source, visualizer_config

def open_speck2e():
    return samna.device.open_device("Speck2eDevKit:0")

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
        [dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"]
    )

    streamer.set_streamer_endpoint("tcp://0.0.0.0:40000")
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {"tcp://0.0.0.0:40000"} fails')

    return streamer

devkit_name = "speck2edevkit:0"
streamer_endpoint = "tcp://0.0.0.0:40000"
gui_process = open_visualizer(streamer_endpoint, 0.75, 0.75)
dynapcnn_net.to(device=devkit_name,monitor_layers=["dvs", -1],  # Last layer
                    chip_layers_ordering="auto",)

# here directly taking samna_config and later putting applying configuration
config = dynapcnn_net.samna_config
lyrs = dynapcnn_net.chip_layers_ordering[-1]
# print(lyrs)
config.dvs_layer.monitor_enable = True
config.cnn_layers[lyrs].monitor_enable = True


dk = open_speck2e()
dk.get_model().apply_configuration(config)


graph = samna.graph.EventFilterGraph()
streamer = build_samna_event_route(graph, dk)

(source ,readout_spike, spike_collection_filter, spike_count_filter,_) = graph.sequential(
                [
                    dk.get_model_source_node(),
                    "Speck2eOutputMemberSelect",
                    "Speck2eSpikeCollectionNode",
                    "Speck2eSpikeCountNode",
                    streamer,
                ]
        )

spike_collection_filter.set_interval_milli_sec(10)
readout_spike.set_white_list([lyrs], "layer")
spike_count_filter.set_feature_count(feature_count)

_, readout_filter, _ = graph.sequential([spike_collection_filter, "Speck2eCustomFilterNode", streamer])
readout_filter.set_filter_function(custom_readout)

# power details
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
threading.Thread(target=inference, daemon=True).start()

gui_process.join()

ps = get_events()

readout_filter.stop()
graph.stop()

# import pickle
# # with open('Optical_Flow_tinycmax/data/collection1_custom64_lr.npy', 'wb') as f:
# #     pickle.dump(collection0, f)

# with open('Optical_Flow_tinycmax/data/collection2_flowmaps64_depth.npy', 'wb') as f:
#     pickle.dump(collection2, f)


channel_data = [[] for _ in range(5)]
for record in ps:
    #print(record)
    channel_data[record.channel].append((record.timestamp, record.value))

output_path = "Optical_Flow_tinycmax/data/power_data.npy"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

numpy_arrays = [np.array(data) for data in channel_data]
np.save(output_path, numpy_arrays)
