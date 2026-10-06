# Collect events from Speck (x, y, t, p) and save to CSV in real time

from pathlib import Path
# Recordings go to speck_inference/data_optical_flow
DATA_DIR = Path(__file__).resolve().parent.parent / "data_optical_flow"

import sinabs
import sinabs.backend.dynapcnn as sio
import os
import time
import torch
import imageio
import threading
import samna
import samnagui
import multiprocessing
import numpy as np
import csv
import queue
import atexit
import psutil
import json

class EventCollection:
    def __init__(self, buffer_size=50000, flush_interval=0.01):
        self.img_folder = str(DATA_DIR)
        self.infer_count = 0
        self.event_queue = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.running = True        # controls flush thread
        self.collecting = True     # controls collector thread
        self.graph_stopped = False # set after graph.stop() so collector drains sink before exiting
        self.csv_lock = threading.Lock()  # prevents concurrent CSV writes

        # Start a background thread for periodic flushing
        self.flush_thread = threading.Thread(target=self.periodic_flush, daemon=True)
        self.flush_thread.start()

        # Ensure flushing on exit
        atexit.register(self.stop)

    def open_speck2f(self):
        return samna.device.open_device("Speck2fDevKit:0")

    def build_samna_event_route(self, dk, dvs_graph, streamer_endpoint):
        _, _, streamer = dvs_graph.sequential(
            [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
        )

        config_source, _ = dvs_graph.sequential(
            [samna.BasicSourceNode_ui_event(), streamer]
        )

        streamer.set_streamer_endpoint(streamer_endpoint)
        if streamer.wait_for_receiver_count() == 0:
            raise Exception(f"Connecting to visualizer on {streamer_endpoint} failed")

        return config_source

    def open_visualizer(self, receiver_endpoint, window_width, window_height):
        gui_process = multiprocessing.Process(
            target=samnagui.run_visualizer,
            args=(receiver_endpoint, window_width, window_height),
        )
        gui_process.start()
        return gui_process

    def event_collector(self, gui_process, sink):
        """Collects events from Speck and stores them in the queue for writing."""
        # MS: time.sleep(0.5) — too long, was dropping real early events; reduced to 0.1s
        # Drain stale events buffered before device was fully configured
        time.sleep(0.1)
        sink.get_events()
        print("Sink buffer cleared, starting clean collection.")

        # MS: while gui_process.is_alive():
        # MS: used self.collecting flag set before join — collector exited mid-sleep, missing
        # buffered sink events. Now: drain until sink empty after graph_stopped signal.
        try:
            while True:
                events_batch = sink.get_events()
                if events_batch:
                    for event in events_batch:
                        if event.feature in (0, 1):
                            try:
                                self.event_queue.put(
                                    [int(event.x), int(event.y), int(event.timestamp), int(event.feature)],
                                    block=True, timeout=1.0
                                )
                            except queue.Full:
                                print("[WARN] Queue still full after 1s, dropping event")
                elif self.graph_stopped:
                    # Sink is empty AND graph has stopped — no more events will ever arrive
                    break
                time.sleep(0.01)
        except Exception as e:
            print(f"[ERROR] event_collector crashed: {e}")

    def flush_event_queue(self):
        """Drains the queue and appends all events to the CSV file."""
        if self.event_queue.empty():
            return

        events_to_flush = []
        while not self.event_queue.empty():
            events_to_flush.append(self.event_queue.get())

        events_to_flush.sort(key=lambda e: e[2])  # sort by timestamp

        csv_file_path = f'{self.img_folder}/events_check4.csv'
        write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0

        with self.csv_lock:
            with open(csv_file_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                if write_header:
                    csv_writer.writerow(['x', 'y', 't', 'p'])
                csv_writer.writerows(events_to_flush)

        print(f"Flushed {len(events_to_flush)} events to CSV.")

    def periodic_flush(self):
        """Runs in a background thread and periodically flushes events to CSV."""
        try:
            while self.running:
                time.sleep(self.flush_interval)
                self.flush_event_queue()
        except Exception as e:
            print(f"[ERROR] periodic_flush crashed: {e}")

    def stop(self):
        """Stops event collection and writes any remaining data to CSV."""
        self.collecting = False
        # Drain the queue fully before stopping the flush thread
        # MS: running=False was set first — flush thread stopped before queue was drained
        while not self.event_queue.empty():
            self.flush_event_queue()
            time.sleep(0.01)
        self.running = False
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=2)
        # Final flush to catch any last stragglers
        self.flush_event_queue()

    def start(self):
        os.makedirs(self.img_folder, exist_ok=True)

        streamer_endpoint = "tcp://0.0.0.0:40000"

        gui_process = self.open_visualizer(streamer_endpoint, 1, 1)

        dk = self.open_speck2f()
        stopWatch = dk.get_stop_watch()
        t_start_wallclock = time.time()  # wall-clock anchor: align DVS ticks to this absolute time
        stopWatch.start()
        metadata = {
            "t_start_wallclock": t_start_wallclock,       # seconds since epoch (time.time())
            "t_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start_wallclock)),
            "note": "DVS event timestamps are hardware ticks from stopWatch.start(). "
                    "Use t_start_wallclock to align with OptiTrack: "
                    "absolute_time = t_start_wallclock + dvs_tick * tick_duration_seconds"
        }
        metadata_path = f'{self.img_folder}/recording_metadata4.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
        dk_io = dk.get_io_module()
        dk_io.set_slow_clk_rate(10)
        dk_io.set_slow_clk(True)
        graph = samna.graph.EventFilterGraph()
        config_source = self.build_samna_event_route(dk, graph, streamer_endpoint)

        sink = samna.graph.sink_from(dk.get_model().get_source_node())

        visualizer_config = samna.ui.VisualizerConfiguration(
            plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1, 1])]
        )
        config_source.write([visualizer_config])

        config = samna.speck2f.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_enable = True
        config.dvs_filter.enable = True
        # MS: config.dvs_layer.mirror.x = False  — lens naturally inverts x, keep hardware default (True)
        dk.get_model().apply_configuration(config)

        collector_thread = threading.Thread(target=self.event_collector, args=(gui_process, sink))
        collector_thread.start()

        graph.start()

        # MS: snapshot children once at t+1s — missed children spawned later, close not detected
        # Now: continuously discover new children each iteration while parent is alive
        gui_pids = {gui_process.pid}
        recording_start = time.time()
        print("Recording... Close the visualizer window to stop.")

        try:
            while True:
                time.sleep(0.5)

                # Keep adding newly spawned children while the parent is still alive
                if gui_process.is_alive():
                    try:
                        new_children = {c.pid for c in psutil.Process(gui_process.pid).children(recursive=True)}
                        gui_pids |= new_children
                    except psutil.NoSuchProcess:
                        pass

                # Allow 3s startup grace before checking for exit (parent exits early during init)
                if time.time() - recording_start > 3:
                    alive = [p for p in gui_pids if psutil.pid_exists(p)
                             and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]
                    if not alive:
                        print("Visualizer closed, stopping collection...")
                        break
        except KeyboardInterrupt:
            print("Stopping collection...")

        graph.stop()
        self.graph_stopped = True  # collector drains sink until empty, then exits
        print("Waiting for collector to finish draining sink...")
        collector_thread.join(timeout=30)
        # Collector is done — no new events will enter the queue
        # self.stop() sets running=False, joins flush thread, then calls flush_event_queue()
        # which drains everything remaining in one shot
        self.stop()
        t_end_wallclock = time.time()

        # Measure tick duration empirically: read CSV and compare timestamp range to wall-clock duration
        csv_file_path = f'{self.img_folder}/events_check3.csv'
        try:
            import pandas as pd
            df = pd.read_csv(csv_file_path)
            if len(df) > 1:
                dvs_duration_ticks = df["t"].iloc[-1] - df["t"].iloc[0]
                wallclock_duration  = t_end_wallclock - t_start_wallclock
                tick_duration_sec   = wallclock_duration / dvs_duration_ticks
                print(f"Measured tick duration: {tick_duration_sec*1e6:.4f} µs/tick  "
                      f"({dvs_duration_ticks} ticks over {wallclock_duration:.2f}s)")
                metadata["tick_duration_seconds"] = tick_duration_sec
                metadata["t_end_wallclock"] = t_end_wallclock
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not measure tick duration: {e}")

        samna.device.close_device(dk)

        print("Event collection stopped successfully.")
        gui_process.terminate()

# Run the event collector
event_collector = EventCollection()
event_collector.start()
