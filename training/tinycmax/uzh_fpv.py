from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from pathlib import Path

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

from tinycmax.data_utils import (
    batched,
    ConcatBatchSampler,
    InfiniteDataLoader,
    only_add_batch_dim,
    time_first_collate,
)


@dataclass
class UzhFpvSequence:
    root_dir: Path
    recording: str
    time_window: float | int | None
    count_window: int | None
    chunk_size: int = 100
    seq_len: int | None = None
    time: tuple[float, float] | tuple[int, int] | None = None  # start, end
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    rectify: bool = False
    augmentations: list[str] | None = None

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
        # self.fw_rect_map = self.fs["fw_rect_map"][:]
        # self.bw_rect_map = self.fs["bw_rect_map"][:]

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
        return len(self.chunk_map)

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


class UzhFpvDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        time_window,
        count_window,
        train_seq_len,
        train_crop,
        train_recordings,
        val_crop,
        val_recordings,
        rectify,
        augmentations,
        batch_size,
        shuffle,
        num_workers,
        download,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.time_window = time_window
        self.count_window = count_window
        self.train_seq_len = train_seq_len
        self.train_crop = train_crop
        self.train_recordings = train_recordings
        self.val_crop = val_crop
        self.val_recordings = val_recordings
        self.rectify = rectify
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.download = download

    def setup(self, stage):
        if stage == "fit":
            train_sequence = partial(
                UzhFpvSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_window=self.count_window,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                rectify=self.rectify,
                augmentations=self.augmentations,
            )
            train_recordings = []
            for rec in self.train_recordings:
                if isinstance(rec, str):
                    rec = (rec, None)
                r, t = rec
                seq = train_sequence(recording=r, time=t)
                train_recordings.extend([rec] * int(seq.rec_duration / seq.seq_duration))
            self.train_dataset = ConcatDataset([train_sequence(recording=r, time=t) for r, t in train_recordings])
            self.train_frame_shape = (self.batch_size, 2, *self.train_dataset.datasets[0].frame_shape)

        if stage in ["fit", "validate"]:
            val_sequence = partial(
                UzhFpvSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_window=self.count_window,
                crop=self.val_crop,
                rectify=self.rectify,
            )
            for i, rec in enumerate(self.val_recordings):
                if isinstance(rec, str):
                    rec = (rec, None)
                self.val_recordings[i] = rec
                print(f"Processing recording: {self.val_recordings[i]}")  # to remove
            self.val_dataset = ConcatDataset([val_sequence(recording=r, time=t) for r, t in self.val_recordings])
            total_samples = len(self.val_dataset)
            print(f"Total validation samples: {total_samples}")
            for i, dataset in enumerate(self.val_dataset.datasets):
                print(f"Dataset {i}: {len(dataset)} samples")
                print(f"Dataset {i}: {len(dataset[0])} samples")
                print(f"Dataset type {i}: {type(dataset[0])} samples")

                # Try fetching the first sample to check its shape and type
                example_sample = dataset[0]  # First sample from the dataset

                print(f"Dataset type {i}: {type(example_sample)}")  # Type of sample
                if isinstance(example_sample, DotMap):
                    print(f"Dataset {i} sample keys: {example_sample.keys()}")  # Print available keys

                # Assuming the sample contains an 'events' field (modify if needed)
                if "frames" in example_sample:
                    print(f"Dataset {i} sample 'frames' shape: {example_sample.frames.shape}")

            self.val_frame_shape = (1, 2, *self.val_dataset.datasets[0].frame_shape)
            print(f"Validation frame shape: {self.val_frame_shape}")

    def train_dataloader(self):
        sampler = ConcatBatchSampler(self.train_dataset, self.batch_size, shuffle=self.shuffle)
        return InfiniteDataLoader(
            self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=time_first_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers // 2,
            collate_fn=only_add_batch_dim,
        )


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate

    # get config
    with initialize(config_path="../config/datamodule", version_base=None):
        config = compose(config_name="uzh_fpv", overrides=["download=false"])
        # config = compose(config_name="speck", overrides=["download=false"])

    # download data
    datamodule = instantiate(config)
    datamodule.prepare_data()
