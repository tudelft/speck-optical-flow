import threading
import samna
import samnagui
import multiprocessing
import numpy as np
import csv
import queue
import atexit
import time
import os

class EventCollection:
    def __init__(self, buffer_size=10000, flush_interval=0.01):
        self.img_folder = "/home/manu-singh/Speck_Optical_Flow/data_optical_flow"
        os.makedirs(self.img_folder, exist_ok=True)

        self.event_queue = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.running = True
        self.min_event_threshold = 100
        self.min_event_wait = 2  # seconds

        self.flush_thread = threading.Thread(target=self.periodic_flush, daemon=True)
        self.flush_thread.start()

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
        """Collects events and stops automatically if event rate drops too low."""
        last_check_time = time.time()
        event_counter = 0

        while self.running:
            events_batch = sink.get_events()
            if events_batch:
                for event in events_batch:
                    if event.feature in (0, 1):
                        try:
                            self.event_queue.put_nowait(
                                [int(event.x), int(event.y), int(event.timestamp), int(event.feature)]
                            )
                            event_counter += 1
                        except queue.Full:
                            # Buffer full: flush immediately then retry
                            self.flush_event_queue()
                            self.event_queue.put_nowait(
                                [int(event.x), int(event.y), int(event.timestamp), int(event.feature)]
                            )
                            event_counter += 1

            current_time = time.time()

            if current_time - last_check_time >= self.min_event_wait:
                print(f"[DEBUG] {event_counter} events in last {self.min_event_wait}s")
                if event_counter < self.min_event_threshold:
                    print(f"Stopping: only {event_counter} events received in last {self.min_event_wait}s.")
                    self.running = False
                    break
                event_counter = 0
                last_check_time = current_time

            time.sleep(0.01)

    def flush_event_queue(self):
        if self.event_queue.empty():
            return

        events_to_flush = []
        while not self.event_queue.empty():
            events_to_flush.append(self.event_queue.get())

        events_to_flush.sort(key=lambda e: e[2])  # sort by timestamp

        csv_file_path = f'{self.img_folder}/events_check.csv'
        write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0

        with open(csv_file_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if write_header:
                csv_writer.writerow(['x', 'y', 't', 'p'])
            csv_writer.writerows(events_to_flush)

        print(f"Flushed {len(events_to_flush)} events to CSV.")

    def periodic_flush(self):
        while self.running:
            time.sleep(self.flush_interval)
            if self.event_queue.qsize() >= self.min_event_threshold:
                self.flush_event_queue()

    def stop(self):
        # Always do cleanup regardless of whether running was already False
        self.running = False
        print("Stopping collection and flushing remaining events...")

        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=2)
            if self.flush_thread.is_alive():
                print("Warning: Flush thread did not exit in time.")

        # Final flush — always runs, even when auto-stop triggered
        self.flush_event_queue()

    def start(self):
        streamer_endpoint = "tcp://0.0.0.0:40000"

        gui_process = self.open_visualizer(streamer_endpoint, 1, 1)

        dk = self.open_speck2f()
        stopWatch = dk.get_stop_watch()
        stopWatch.start()

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
        dk.get_model().apply_configuration(config)

        collector_thread = threading.Thread(target=self.event_collector, args=(gui_process, sink))
        collector_thread.start()

        graph.start()

        try:
            while collector_thread.is_alive():
                collector_thread.join(timeout=0.5)

            if gui_process.is_alive():
                print("Terminating GUI process due to low events.")
                gui_process.terminate()
                gui_process.join()

        except KeyboardInterrupt:
            print("Interrupted by user.")
            self.running = False

        collector_thread.join(timeout=5)
        graph.stop()
        self.stop()
        samna.device.close_device(dk)

        print("Event collection stopped successfully.")

# Run the event collector
event_collector = EventCollection()
event_collector.start()
