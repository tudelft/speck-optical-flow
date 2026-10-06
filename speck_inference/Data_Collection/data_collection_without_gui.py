from pathlib import Path
# Recordings go to speck_inference/data_optical_flow
DATA_DIR = Path(__file__).resolve().parent.parent / "data_optical_flow"

import threading
import samna
import numpy as np
import csv
import queue
import atexit
import time
import os

class EventCollection:
    def __init__(self, buffer_size=10000, flush_interval=0.02, use_gui=False):
        self.img_folder = str(DATA_DIR)
        os.makedirs(self.img_folder, exist_ok=True)

        self.event_queue = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.running = True
        self.min_event_threshold = 100
        self.min_event_wait = 2  # seconds

        self.flush_thread = threading.Thread(target=self.periodic_flush, daemon=True)
        self.flush_thread.start()

        self.use_gui = use_gui
        atexit.register(self.stop)

    def open_speck2e(self):
        return samna.device.open_device("Speck2eDevKit:0")

    def build_samna_event_route(self, dk, dvs_graph):
        if self.use_gui:
            _, _, streamer = dvs_graph.sequential(
                [dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"]
            )

            config_source, _ = dvs_graph.sequential(
                [samna.BasicSourceNode_ui_event(), streamer]
            )

            streamer.set_streamer_destination("tcp://0.0.0.0:40000")
            if streamer.wait_for_receiver_count() == 0:
                raise Exception("Connecting to visualizer failed")

            return config_source
        else:
            return None

    def event_collector(self, sink):
        last_check_time = time.time()
        event_counter = 0

        while self.running:
            events_batch = sink.get_events()
            if events_batch:
                for event in events_batch:
                    if event.feature in (0, 1):
                        self.event_queue.put([int(event.x), int(event.y), int(event.timestamp), int(event.feature)])
                        event_counter += 1

            current_time = time.time()
            print(f"[DEBUG] Time since last check: {current_time - last_check_time:.2f}s, Event count: {event_counter}")

            if current_time - last_check_time >= self.min_event_wait:
                if event_counter < self.min_event_threshold:
                    print(f"Stopping: Only {event_counter} events received in last {self.min_event_wait} seconds.")
                    self.running = False
                    break
                else:
                    event_counter = 0
                    last_check_time = current_time

            if self.event_queue.qsize() >= self.buffer_size:
                self.flush_event_queue()
                self.clear_event_queue()

            time.sleep(0.1)

    def flush_event_queue(self):
        if self.event_queue.empty():
            return

        events_to_flush = []
        while not self.event_queue.empty():
            events_to_flush.append(self.event_queue.get())

        events_to_flush.sort(key=lambda e: e[2])

        csv_file_path = f'{self.img_folder}/events_multi11.csv'
        write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0

        with open(csv_file_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if write_header:
                csv_writer.writerow(['x', 'y', 't', 'p'])
            csv_writer.writerows(events_to_flush)

        print(f"Flushed {len(events_to_flush)} events to CSV.")

    def clear_event_queue(self):
        while not self.event_queue.empty():
            self.event_queue.get()

    def periodic_flush(self):
        while self.running:
            time.sleep(self.flush_interval)
            if self.event_queue.qsize() >= self.min_event_threshold:
                self.flush_event_queue()

    def stop(self):
        if not self.running:
            return
        print("Stopping collection and flushing remaining events...")
        self.running = False

        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=2)
            if self.flush_thread.is_alive():
                print("Warning: Flushing thread did not exit in time.")

        self.flush_event_queue()

    def start(self):
        dk = self.open_speck2e()
        stopWatch = dk.get_stop_watch()
        stopWatch.set_enable_value(True)

        dk_io = dk.get_io_module()
        dk_io.set_slow_clk_rate(10)
        dk_io.set_slow_clk(True)

        graph = samna.graph.EventFilterGraph()
        config_source = self.build_samna_event_route(dk, graph)

        sink = samna.graph.sink_from(dk.get_model().get_source_node())

        if self.use_gui and config_source:
            visualizer_config = samna.ui.VisualizerConfiguration(
                plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1, 1])]
            )
            config_source.write([visualizer_config])

        config = samna.speck2e.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_enable = True
        config.dvs_filter.enable = True
        dk.get_model().apply_configuration(config)

        collector_thread = threading.Thread(target=self.event_collector, args=(sink,))
        collector_thread.start()

        graph.start()

        try:
            while self.running and collector_thread.is_alive():
                collector_thread.join(timeout=0.5)

        except KeyboardInterrupt:
            print("Interrupted by user.")
            self.running = False

        finally:
            collector_thread.join(timeout=5)
            graph.stop()
            self.stop()
            samna.device.close_device(dk)
            print("Event collection stopped successfully.")

# Run without GUI
if __name__ == "__main__":
    event_collector = EventCollection(use_gui=False)
    event_collector.start()