# To create list of events gathered by speck and save as csv as well with queue and flush

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

class EventCollection:
    def __init__(self, buffer_size=10000, flush_interval=0.02):
        self.img_folder = "/home/manu/Desktop/SPECK/Visualization/data_optical_flow"
        self.infer_count = 0
        self.event_queue = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size  
        self.flush_interval = flush_interval  
        self.running = True

        # Start a background thread for periodic flushing
        self.flush_thread = threading.Thread(target=self.periodic_flush, daemon=True)
        self.flush_thread.start()

        # Ensure flushing on exit
        atexit.register(self.stop)

    def open_speck2e(self):
        return samna.device.open_device("Speck2eDevKit:0")
    
    def build_samna_event_route(self, dk, dvs_graph, streamer_endpoint):
        _, _, streamer = dvs_graph.sequential(
            [dk.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"]
        )

        config_source, _ = dvs_graph.sequential(
            [samna.BasicSourceNode_ui_event(), streamer]
        )

        streamer.set_streamer_destination(streamer_endpoint)
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
        """Collects events and stores them in a buffer for later writing."""

        while gui_process.is_alive():
            events_batch = sink.get_events()
            if events_batch:
                for event in events_batch:
                    if event.feature in (0, 1):
                        self.event_queue.put([int(event.x), int(event.y), int(event.timestamp), int(event.feature)])


            # Flush if buffer reaches capacity
            if self.event_queue.qsize() >= self.buffer_size:
                self.flush_to_csv()  
                self.clear_event_queue()  # Clear the queue after flushing

            time.sleep(0.1) 

    

    def flush_event_queue(self):
        if self.event_queue.empty():
            return

        events_to_flush = []
        while not self.event_queue.empty():
            events_to_flush.append(self.event_queue.get())

        # Ensure timestamp order
        events_to_flush.sort(key=lambda e: e[2])

        # csv_file_path = f'{self.img_folder}/events_5_check.csv'
        csv_file_path = f'{self.img_folder}/events_simple11.csv'

        with open(csv_file_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) == 0:
                csv_writer.writerow(['x', 'y', 't', 'p'])

            csv_writer.writerows(events_to_flush)

        print(f"Flushed {len(events_to_flush)} events to CSV.")

    def clear_event_queue(self):
        """Clears the event queue after flushing."""
        while not self.event_queue.empty():
            self.event_queue.get()

    def periodic_flush(self):
        """Runs in a background thread and periodically flushes events."""
        while self.running:
            time.sleep(self.flush_interval)
            if not self.running:  # Double-check before flushing
                break
            self.flush_event_queue()

    def stop(self):
        """Stops event collection and writes any remaining data to CSV."""
        self.running = False
        print("hello")
        if self.flush_thread.is_alive():
            print("Waiting for flushing thread to stop...")
            self.flush_thread.join(timeout=2)  # Prevent hanging
            if self.flush_thread.is_alive():
                print("Warning: Flushing thread did not exit in time.")
        self.flush_event_queue()

    def start(self):
        streamer_endpoint = "tcp://0.0.0.0:40000"
        
        gui_process = self.open_visualizer(streamer_endpoint, 1, 1)
        
        dk = self.open_speck2e()
        stopWatch = dk.get_stop_watch()
        stopWatch.set_enable_value(True)
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

        config = samna.speck2e.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_enable = True
        config.dvs_filter.enable = True
        dk.get_model().apply_configuration(config)

        collector_thread = threading.Thread(target=self.event_collector, args=(gui_process, sink))
        collector_thread.start()

        graph.start()

        gui_process.join()

        graph.stop()
        
        collector_thread.join(timeout=5)
        self.stop()  
        samna.device.close_device(dk)

        print("Event collection stopped successfully.")
        gui_process.terminate()

# Run the optimized event collector
event_collector = EventCollection()
event_collector.start()
