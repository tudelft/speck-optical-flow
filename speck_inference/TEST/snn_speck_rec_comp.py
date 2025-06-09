### make patterned data in simple rasters.

import sinabs
import torch
import torch.nn as nn
from sinabs.from_torch import from_model
import numpy as np
from sinabs.utils import normalize_weights
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
import pickle
import sinabs.layers as sl
import sinabs.activation as sina

#### making data ##############################################################
def generate_hollow_rectangle_rasters_wrap(grid_size=16, rect_size=(6, 8), steps=25):
    """
    Generate 16x16 binary rasters with a hollow rectangle moving to the right, wrapping around the grid.

    Args:
        grid_size (int): Size of the square grid.
        rect_size (tuple): (height, width) of the rectangle.
        steps (int): Number of time steps (horizontal positions).

    Returns:
        List of 2D numpy arrays (grid_size x grid_size).
    """
    rasters = []
    rect_h, rect_w = rect_size
    y_start = (grid_size - rect_h) // 2

    for step in range(steps):
        raster = np.zeros((grid_size, grid_size), dtype=np.uint8)
        x_start = step % grid_size  # wrap around

        for dy in [0, rect_h - 1]:  # top and bottom edges
            for dx in range(rect_w):
                x = (x_start + dx) % grid_size
                y = y_start + dy
                if 0 <= y < grid_size:
                    raster[y, x] = 1

        for dx in [0, rect_w - 1]:  # left and right edges
            for dy in range(rect_h):
                x = (x_start + dx) % grid_size
                y = y_start + dy
                if 0 <= y < grid_size:
                    raster[y, x] = 1

        rasters.append(torch.tensor(raster).unsqueeze(0).unsqueeze(0))

    return rasters


# Example usage: generate and combine for visualization
hollow_rasters = generate_hollow_rectangle_rasters_wrap()

with open('Optical_Flow_tinycmax/data/hollow_rasters.npy', 'wb') as f: # all layers normalised with scaling in first layer and lower mem first layer.
    pickle.dump(hollow_rasters, f)

print(type(hollow_rasters))
print(hollow_rasters[0].shape)
dt = 1e-3  # 1 millisecond time step
truncate = False  # Don't limit the number of spikes
delay_factor = 0


# ###############################################################################

# speck_ann = nn.Sequential(
#     nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=2, bias=False),
#     nn.ReLU(),
 
#     nn.Conv2d(2, 2, kernel_size=3, padding=1,stride=2, bias=False),
#     nn.ReLU(),

#     nn.Conv2d(2, 2, kernel_size=3, padding=1,stride=1, bias=False),
#     nn.ReLU(),

#     nn.Conv2d(2, 2, kernel_size=3, padding=1,stride=1, bias=False),
#     nn.ReLU(),

# )

# weight_value = 0.1
# speck_ann[0].weight.data = torch.ones_like(speck_ann[0].weight.data) * weight_value
# speck_ann[2].weight.data = torch.ones_like(speck_ann[2].weight.data) * weight_value
# speck_ann[4].weight.data = torch.ones_like(speck_ann[4].weight.data) * weight_value
# speck_ann[6].weight.data = torch.ones_like(speck_ann[6].weight.data) * weight_value


# speck_ann = speck_ann.to("cpu")

# sinabs_model = sinabs.from_model(speck_ann, add_spiking_output=True, batch_size=1).spiking_model.cpu()
# print(sinabs_model)

# with torch.no_grad():
#     for name, param in sinabs_model.named_parameters():
            

#         if name == '0.weight':  
#             param.data *= 1   
#             param_data0 = param.data

#         if name == '2.weight':
#             param.data *= 1
#             param_data2 = param.data

#         if name == '4.weight':  
#             param.data *= 1  
#             param_data4 = param.data

#         if name == '6.weight':
#             param.data *= 1
#             param_data6 = param.data


# print(np.amax(np.array(param_data0)))
# print(np.amax(np.array(param_data2)))
# print(np.amax(np.array(param_data4)))
# print(np.amax(np.array(param_data6)))



# dynapcnn_net = DynapcnnNetwork(
#     snn=sinabs_model, 
#     input_shape=(1, 16, 16), 
#     discretize=True, 
#     dvs_input=False)

# devkit_name = "speck2edevkit:0"

# samna_cfg = dynapcnn_net.make_config(device=devkit_name)
# samna_cfg.cnn_layers[3].destinations[1].enable = True
# samna_cfg.cnn_layers[3].destinations[1].layer = 3

# dynapcnn_net.to(device=devkit_name, 
#             monitor_layers=[-1], 
#             chip_layers_ordering="auto",
#             ) 
# print(f"The SNN is deployed on the core: {dynapcnn_net.chip_layers_ordering}")

# for layer in [0, 1, 2, 3]:
#     print(f"Is layer {layer} output turned on: {samna_cfg.cnn_layers[layer].destinations[0].enable}")
#     print(f"The destination layer of layer {layer} is layer {samna_cfg.cnn_layers[layer].destinations[0].layer}")
    
#     print(f"Is layer {layer} output turned on: {samna_cfg.cnn_layers[layer].destinations[1].enable}")
#     print(f"The destination layer of layer {layer} is layer {samna_cfg.cnn_layers[layer].destinations[1].layer}")
#     print("--")

# print(dynapcnn_net)

# chip_factory = ChipFactory(devkit_name)
# layer_in = dynapcnn_net.chip_layers_ordering[0]

# output_list = []

# for i in range(len(hollow_rasters)):

#     input_tensor = chip_factory.raster_to_events(raster = hollow_rasters[i], layer=layer_in, dt=dt, truncate=truncate, delay_factor=delay_factor)
#     # print(len(input_tensor))

#     output_events = dynapcnn_net(input_tensor)
#     print(len(output_events))

#     #output_list.append(output_events)

#     if len(output_events) > 0:
#         output_raster = ChipFactory.events_to_raster_fast(events=output_events,
#                                           shape=(2,4,4))
#         output_list.append(output_raster)

#     else:
#         output_list.append(torch.zeros((1, 2, 4, 4)))


# with open('Optical_Flow_tinycmax/data/speck_rec_results4_4.npy', 'wb') as f: # all layers normalised with scaling in first layer and lower mem first layer.
#     pickle.dump(output_list, f)

#############################################################################

#### SNN recurrency #############################



# class RecurrentSNNNet(nn.Module):
#     def __init__(self):
#         super(RecurrentSNNNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=2, bias=False)
#         self.iaf1 = sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0,
#                           surrogate_grad_fn=sina.SingleExponential(0.5, 1.0))

#         self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1, stride=2, bias=False)
#         self.iaf2 = sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0,
#                           surrogate_grad_fn=sina.SingleExponential(0.5, 1.0))

#         self.conv3 = nn.Conv2d(2, 2, kernel_size=3, padding=1, stride=1, bias=False)
#         self.iaf3 = sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0,
#                           surrogate_grad_fn=sina.SingleExponential(0.5, 1.0))

#         self.conv4 = nn.Conv2d(2, 2, kernel_size=3, padding=1, stride=1, bias=False)
#         self.iaf4 = sl.IAFSqueeze(batch_size=1, min_v_mem=-1.0,
#                           surrogate_grad_fn=sina.SingleExponential(0.5, 1.0))

#         # Initialize weights to 0.1
#         nn.init.constant_(self.conv1.weight, 0.1)
#         nn.init.constant_(self.conv2.weight, 0.1)
#         nn.init.constant_(self.conv3.weight, 0.1)
#         nn.init.constant_(self.conv4.weight, 0.1)

#     def forward(self, x, prev_out):
#         x = self.conv1(x)
#         x = self.iaf1(x)

#         x = self.conv2(x)
#         x = self.iaf2(x)

#          # initialize if no previous output
#         if prev_out is None:
#             prev_out = torch.zeros_like(x) 

#         x = self.conv3(x) # + prev_out
#         x = self.iaf3(x)
        

#         # layer reconnect to layer 4
#         out = self.conv4(x + prev_out)
#         out = self.iaf4(out)

#         return out
    

# model = RecurrentSNNNet()  # using the model from earlier
# prev_out = None
# snn_output = []

# for t in range(len(hollow_rasters)):

#     with torch.no_grad():
#         x = hollow_rasters[t]
#         x = x.float() 
#         out = model(x, prev_out)
#         print(out.shape)
#         #print(out.shape)
#         prev_out = out.detach()

#         snn_output.append(out)


# with open('Optical_Flow_tinycmax/data/snn_rec_results4_4.npy', 'wb') as f: # all layers normalised with scaling in first layer and lower mem first layer.
#     pickle.dump(snn_output, f)