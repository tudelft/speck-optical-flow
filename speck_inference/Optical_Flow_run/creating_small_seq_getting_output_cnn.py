from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import torch.nn as nn

import cv2
from dotmap import DotMap
import h5py
import hdf5plugin
from lightning import LightningDataModule
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from rich.progress import track
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive
import yaml
from sinabs.layers import IAFSqueeze

from data_utils import (
    batched,
    ConcatBatchSampler,
    InfiniteDataLoader,
    only_add_batch_dim,
    time_first_collate,
)



class UzhFpvSequence:
    def __init__(self, 
                 root_dir: Path,
                 recording: str,
                 time_window: float | int | None,
                 count_window: int | None = None, 
                 chunk_size: int = 100,
                 seq_len: int | None = None,
                 time: tuple[float, float] | tuple[int, int] | None = None,
                 crop: tuple[int, ...] | None = None,
                 rectify: bool = False,
                 augmentations: list[str] | None = None):

        
        self.root_dir = root_dir
        self.recording = recording
        self.time_window = time_window
        self.count_window = count_window
        self.chunk_size = chunk_size
        self.seq_len = seq_len
        self.time = time
        self.crop = crop
        self.rectify = rectify
        self.augmentations = augmentations

        self.frame_shape = (128, 128)

        self.__post_init__()

    def __post_init__(self):
        # checks
        # TODO: implement count window
        assert not (self.time_window is not None and self.count_window is not None)
        assert self.count_window is None

        # data
        # open large h5 files only once
        self.fs = h5py.File(self.root_dir / f"{self.recording}.h5", "r")

        # attributes
        self.sensor_size = tuple(self.fs.attrs["sensor_size"])  # height, width
        #self.fw_rect_map = self.fs["fw_rect_map"][:]
        #self.bw_rect_map = self.fs["bw_rect_map"][:]

        # get duration of recording
        self.t0, self.tk = self.fs["events/t"][[0, -1]]
        if self.time is not None:
            t0, tk = self.time
            t0 = t0 + self.t0 if t0 is not None else self.t0
            tk = tk + self.t0 if tk is not None else self.tk
            self.t0, self.tk = t0, tk
        self.rec_duration = self.tk - self.t0

        # slice dataset, pre-compute crop, pre-compute augmentation
        self.reset()

        # set frame shape
        self.frame_shape = (
            self.crop_corners[2] - self.crop_corners[0],
            self.crop_corners[3] - self.crop_corners[1],
        )

        # mapping from chunks to steps
        # match seq_len if given
        self.chunk_size = self.seq_len if self.seq_len is not None else self.chunk_size
        self.chunk_map = batched(range(len(self.t_start)), self.chunk_size)

    def init_slice(self):
        # start and end time
        if self.seq_len is not None:  # randomly-sliced sequence of seq_len
            start, end = self.t0, max(self.t0, self.tk - (self.seq_len + 1) * self.time_window)  # only full
            if np.issubdtype(self.fs["events/t"], np.integer):
                t_start = np.random.randint(start, max(start + 1, end))
            elif np.issubdtype(self.fs["events/t"], np.floating):
                t_start = np.random.uniform(start, end)
            n_full_windows = self.seq_len
        else:  # full sequence
            t_start, t_end = self.t0, self.tk
            n_full_windows = max(1, int((t_end - t_start) / self.time_window))  # at least 1 window

        # window making
        # use linspace because could be floats (no rounding errors?)
        linspace = np.linspace(t_start, n_full_windows * self.time_window + t_start, n_full_windows + 1)
        self.t_start, self.t_end = linspace[:-1], linspace[1:]
        self.seq_duration = n_full_windows * self.time_window

    def init_crop(self):
        if self.crop:
            if len(self.crop) == 2:  # height, width
                h, w = self.crop
                top = np.random.randint(self.sensor_size[0] - h + 1)  # +1 because exclusive
                left = np.random.randint(self.sensor_size[1] - w + 1)
                self.crop_corners = (top, left, top + h, left + w)
            elif len(self.crop) == 4:  # top, left, bottom, right
                self.crop_corners = self.crop
        else:
            self.crop_corners = (0, 0, *self.sensor_size)

    def init_augmentation(self):
        self.augmentation = []
        if self.augmentations is not None:
            for aug in self.augmentations:
                if np.random.rand() < 0.5:
                    self.augmentation.append(aug)

    def reset(self):
        self.init_slice()  # slice up dataset
        self.init_crop()  # pre-compute crop
        self.init_augmentation()  # pre-compute augmentation

    def __len__(self):
        return len(self.chunk_map) #100 This is probably wrong. 

    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # get chunk
        chunk = self.chunk_map[idx]

        # go over slices
        frames, auxs, targets = [], DotMap(), DotMap()
        for i in chunk:
            # convert to indices
            start = bisect_left(self.fs["events/t"], self.t_start[i])
            end = bisect_left(self.fs["events/t"], self.t_end[i])

            # get events as list
            t = self.fs["events/t"][start:end]  # uint32
            y = self.fs["events/y"][start:end]  # uint16
            x = self.fs["events/x"][start:end]  # uint16
            p = self.fs["events/p"][start:end]  # uint8 in {0, 1}

            # rectify list: forward rectification
            if self.rectify:
                x_rect, y_rect = self.fw_rect_map[y.astype(np.int64), x.astype(np.int64)].T
            else:
                x_rect, y_rect = x, y

            # list of events to structured array
            dtype = np.dtype([("t", np.float64), ("y", np.float32), ("x", np.float32), ("p", np.int8)])
            lst = np.empty(len(t), dtype=dtype)
            lst["t"] = t
            lst["y"] = y_rect
            lst["x"] = x_rect
            lst["p"] = p

            # crop list
            top, left, bottom, right = self.crop_corners
            mask = (y_rect >= top) & (y_rect < bottom) & (x_rect >= left) & (x_rect < right)
            lst = lst[mask]
            lst["y"] -= top
            lst["x"] -= left

            # make into event count frame
            # use unrectified coordinates
            y = torch.from_numpy(y.astype(np.int64))
            x = torch.from_numpy(x.astype(np.int64))
            p = torch.from_numpy(p.astype(np.int64))
            frame = torch.zeros(2, *self.sensor_size, dtype=torch.int64)  # torch is faster
            frame.index_put_((p, y, x), torch.ones_like(p), accumulate=True)

            # rectify frame: backward rectification
            # backward to prevent lines in frames
            if self.rectify:
                frame = cv2.remap(
                    frame.numpy().transpose(1, 2, 0), self.bw_rect_map, None, interpolation=cv2.INTER_NEAREST
                )
                frame = torch.from_numpy(frame.transpose(2, 0, 1))

            # crop frame
            frame = frame[..., top:bottom, left:right]

            # discard if few events or same timestamp
            if len(lst) < 10 or lst["t"][-1] == lst["t"][0]:
                lst = np.array([], dtype=lst.dtype)
                frame = torch.zeros_like(frame)

            # format list of events: normalize time, polarity to {-1, 1}
            # after cropping, else normalized timestamp not correct
            lst["t"] = (lst["t"] - lst["t"][0]) / (lst["t"][-1] - lst["t"][0]) if len(lst) else lst["t"]
            lst["p"] = lst["p"] * 2 - 1

            # append
            frames.append(frame)
            auxs.events += [lst]
            auxs.counts += [len(lst)]

        # stack and pad
        frames = torch.stack(frames)
        max_len = max(auxs.counts)
        auxs.events = rfn.structured_to_unstructured(
            np.stack([np.pad(e, (0, max_len - len(e))) for e in auxs.events]), dtype=np.float32
        )
        auxs = DotMap({k: torch.tensor(v) for k, v in auxs.items()}, _dynamic=False)  # convert to static dotmap
        targets = DotMap({k: torch.stack(v) for k, v in targets.items()}, _dynamic=False)

        # apply augmentations; more efficient on chunks
        # not used with targets, so leave those out
        if "flip_t" in self.augmentation:
            frames = frames.flip(0)
            auxs.events[..., 0] = 1 - auxs.events[..., 0]
            auxs.events = auxs.events.flip(0)
            auxs.counts = auxs.counts.flip(0)
        if "flip_pol" in self.augmentation:
            frames = frames.flip(1)
            auxs.events[..., 3] *= -1
        if "flip_ud" in self.augmentation:
            frames = frames.flip(2)
            auxs.events[..., 1] = (bottom - top - 1) - auxs.events[..., 1]
        if "flip_lr" in self.augmentation:
            frames = frames.flip(3)
            auxs.events[..., 2] = (right - left - 2) - auxs.events[..., 2]

        # return static dotmap
        sample = DotMap(
            frames=frames.float(),
            auxs=auxs,
            targets=targets,
            recording=self.recording,
            eofs=[i == len(self.t_start) - 1 for i in chunk],
            _dynamic=False,
        )

        return sample
    
    
val_sequence = partial(
                UzhFpvSequence,
                root_dir= Path("Optical_Flow_tinycmax/data/speck"),
                recording='events_orig3', # events_hand
                time_window=10000,
            )

val_dataset = ConcatDataset([val_sequence()])
val_frame_shape = (1, 2, *val_dataset.datasets[0].frame_shape)

print("val_frame_shape: ", val_frame_shape)
print(type(val_dataset))
print(len(val_dataset))
print(val_dataset)

first_dataset = val_dataset.datasets[0]
print("len of first dataset: ", len(first_dataset))
print(first_dataset[0]['frames'][0].shape)
frame = first_dataset[0]['frames'][0].unsqueeze(0)

# #Calling the blocks

# from blocks import conv_encoder, LazyConvGru, upsample_decoder, LazyConvMinGru


# # enc = conv_encoder(out_channels=64, activation_fn=nn.ReLU)
# # memory = LazyConvGru(out_channels=64, kernel_size=3)
# # decoder = upsample_decoder(64, nn.ReLU, final_bias=True)

# enc = conv_encoder(64, activation_fn=nn.ReLU, padding_mode="reflect")
# memory = LazyConvGru(64, 3, padding_mode="reflect")
# decoder = upsample_decoder(
#             64, activation_fn=nn.ReLU, final_bias=True, padding_mode="reflect", mode="flow"
#         )

# #-------------------------------------------------------
# ## Loading weights
# # enc(torch.randn(1, 2, 128, 128))
# # memory(torch.randn(1, 64, 16, 16), None)
# # decoder(torch.randn(1, 64, 16, 16))

# # checkpoint = torch.load("Optical_Flow_tinycmax/iterative_model_hand_16ch_nobias.pt", map_location="cpu")
# checkpoint = torch.load("Optical_Flow_tinycmax/state_dict.pt", map_location="cpu")


# enc_params = {k: v for k, v in checkpoint.items() if "enc" in k}
# memory_params = {k: v for k, v in checkpoint.items() if "memory" in k}
# decoder_params = {k: v for k, v in checkpoint.items() if "decoder" in k}

# enc.load_state_dict(enc_params, strict=False)
# memory.load_state_dict(memory_params, strict=False)
# decoder.load_state_dict(decoder_params, strict=False)

# enc.eval()  # Ensure model is in eval mode
# memory.eval()
# decoder.eval()
# #----------------------------------------------------------


# tinycmax_final_output = []
# count=0

# output_hidden = None




# # running the model
# ### One frame by frame
# for i in range(8):#15 len(first_dataset)
#     for j in range(len(first_dataset[i]['frames'])):
#         sample = (first_dataset[i]['frames'])
#         frame = sample[j].unsqueeze(0)
#         print(frame)

#         output = enc(frame)
#         # if output_hidden is None:
#         #     output_hidden = torch.zeros_like(output)

#         output_mem = memory(output, output_hidden)
#         print(f"Hidden state change: {torch.norm(output_mem)}")
#         output_hidden = output_mem.detach().clone()
#         output_dec = decoder(output_mem)
#         output_flow = output_dec * 32  # 32 is scaling
#         tinycmax_final_output.append(output_flow.detach())
#         count+=1
#         print("count, ", count)

#         print("Decoder output: ", output_flow.shape)
        


# import pickle

# ## saving flow outputs
# # with open('Optical_Flow_tinycmax/data/tinycmax_outputs_flow_16ch_hand.npy', 'wb') as f:
# #     pickle.dump(tinycmax_final_output, f)

# with open('Optical_Flow_tinycmax/data/tinycmax_outputs_original_states.npy', 'wb') as f:
#     pickle.dump(tinycmax_final_output, f)



#### New way-------------------------------------------

import network

model = network.WrappedFlowNetwork(
    memory_channels=16,
    decoder_channels=16,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="reflect",
    scaling=32
)

state_dict = torch.load("Optical_Flow_tinycmax/iterative_model_hand_16ch.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
print(state_dict.keys())

new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("network.", "")  # strip the prefix
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

output_hidden = None
tinycmax_final_output = []

count = 0

for i in range(len(first_dataset)):  # Or len(first_dataset)
    frames = first_dataset[i]['frames']
    print(frames.shape)
    for j in range(len(frames)):
        frame = frames[j].unsqueeze(0).to(device)  # Shape: [1, C, H, W]

        input_dict = {"events": frame}

        with torch.no_grad():
            output_dict, output_hidden = model(input_dict, output_hidden)
            #print(output_dict)
            flow_map = output_dict["flow"]
            tinycmax_final_output.append(flow_map.cpu())  # Save on CPU
            print("count: ", count)
            count+=1


import pickle
# with open('Optical_Flow_tinycmax/data/rqe_hand_states_ann.npy', 'wb') as f:
#     pickle.dump(tinycmax_final_output, f)

with open('Optical_Flow_tinycmax/data/of_orig3_states_ann.npy', 'wb') as f:
    pickle.dump(tinycmax_final_output, f)

