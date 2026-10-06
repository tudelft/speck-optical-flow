"""
Speck DVS optical flow estimation node.

Runs a spiking neural network on the Speck2f chip to estimate optical flow
and divergence from DVS events. Publishes [flow_x, flow_y, divergence] on
/speck/optical_flow as Float32MultiArray.

If hardware is not connected, runs in dummy mode publishing zeros.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import numpy as np
import os
import threading
import time


class SpeckFlowNode(Node):
    def __init__(self):
        super().__init__('speck_flow_node')

        self.publisher_ = self.create_publisher(
            Float32MultiArray, '/speck/optical_flow', 10)

        # Filter parameter
        self.declare_parameter('sensor_alpha', 0.9)
        # Directory holding qs95vlk2_minGRU_depth1_model.pt / .onnx
        # (speck_inference/Optical_Flow_run in this repository)
        self.declare_parameter('model_dir', os.environ.get(
            'SPECK_MODEL_DIR',
            os.path.expanduser(
                '~/speck-optical-flow/speck_inference/Optical_Flow_run')))

        self.filtered_flow_x = 0.0
        self.filtered_flow_y = 0.0

        self.hardware_available = False
        self._stop_event = threading.Event()

        try:
            self._init_hardware()
            self.hardware_available = True
            self.get_logger().info('Speck hardware initialized successfully.')
            # Start inference thread
            self._inference_thread = threading.Thread(
                target=self._inference_loop, daemon=True)
            self._inference_thread.start()
        except Exception as e:
            self.get_logger().error(f'Speck hardware not found: {e}')
            self.get_logger().warn(
                'Running in dummy mode - publishing zeros at 10Hz')
            self._dummy_timer = self.create_timer(0.1, self._publish_dummy)

    def _init_hardware(self):
        """Initialize Speck2f hardware and SNN model."""
        import sinabs.backend.dynapcnn as sindynapcnn
        import sinabs.from_torch
        import torch
        import torch.nn as nn
        import samna
        from sinabs.from_torch import from_model
        from sinabs.backend.dynapcnn.chip_factory import ChipFactory
        import onnxruntime as ort

        # Store imports for use in inference loop
        self._torch = torch
        self._np = np
        self._samna = samna
        self._ChipFactory = ChipFactory

        # Load model
        model_path = self.get_parameter('model_dir').value
        checkpoint = torch.load(
            f'{model_path}/qs95vlk2_minGRU_depth1_model.pt',
            map_location='cpu')
        self._onnx_session = ort.InferenceSession(
            f'{model_path}/qs95vlk2_minGRU_depth1.onnx',
            providers=['CPUExecutionProvider'])

        # Build ANN for DynapCNN conversion
        ann = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7, padding=3, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(),
        )
        ann[0].weight.data = checkpoint['network.encoder.enc1.synapse.weight']
        ann[2].weight.data = checkpoint['network.encoder.enc2.conv0.synapse.weight']
        ann[4].weight.data = checkpoint['network.encoder.enc2.conv1.synapse.weight']
        ann = ann.to('cpu')

        sinabs_model = sinabs.from_model(
            ann, add_spiking_output=True, batch_size=1).spiking_model.cpu()
        with torch.no_grad():
            for name, param in sinabs_model.named_parameters():
                if 'weight' in name:
                    param.data *= 1

        input_shape = (2, 128, 128)
        dynapcnn_net = sindynapcnn.DynapcnnNetwork(
            sinabs_model.cpu(),
            input_shape=input_shape,
            discretize=True,
            dvs_input=True)

        # Deploy to Speck2f
        devkit_name = 'speck2fdevkit:0'
        dynapcnn_net.to(
            device=devkit_name,
            monitor_layers=['dvs', -1],
            chip_layers_ordering='auto')
        config = dynapcnn_net.samna_config
        lyrs = dynapcnn_net.chip_layers_ordering[-1]
        config.dvs_layer.monitor_enable = True
        config.cnn_layers[lyrs].monitor_enable = True

        self._dk = samna.device.open_device('Speck2fDevKit:0')
        self._dk.get_model().apply_configuration(config)

        # Event filter graph
        self._graph = samna.graph.EventFilterGraph()
        (source, readout_spike, spike_collection_filter, _, _) = \
            self._graph.sequential([
                self._dk.get_model_source_node(),
                'Speck2fOutputMemberSelect',
                'Speck2fSpikeCollectionNode',
                'Speck2fSpikeCountNode',
                'VizEventStreamer'])

        spike_collection_filter.set_interval_milli_sec(10)
        readout_spike.set_white_list([lyrs], 'layer')

        # Shared event collection
        self._collection = []
        self._collection_lock = threading.Lock()

        def custom_readout(collection):
            now = time.time()
            with self._collection_lock:
                self._collection.append({
                    'timestamp': now, 'events': collection})

            def generate_result(feature):
                e = samna.ui.Readout()
                e.feature = feature
                return [e]
            return generate_result(int(0))

        _, self._readout_filter, _ = self._graph.sequential(
            [spike_collection_filter, 'Speck2fCustomFilterNode',
             'VizEventStreamer'])
        self._readout_filter.set_filter_function(custom_readout)
        self._graph.start()

        # State for inference
        self._output_hidden = None
        self._quadrant_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def _get_latest_events(self):
        """Retrieve and clear collected events."""
        with self._collection_lock:
            data = self._collection.copy()
            self._collection.clear()
        combined = []
        for item in data:
            for event in item['events']:
                combined.append((item['timestamp'], event))
        return combined

    def _inference_loop(self):
        """Main inference loop running in a background thread."""
        torch = self._torch

        while not self._stop_event.is_set():
            data = self._get_latest_events()
            if len(data) == 0:
                time.sleep(0.005)
                continue

            timestamps, events = zip(*data)
            raster = self._ChipFactory.events_to_raster_fast(
                events=events, shape=(32, 16, 16))

            # ONNX inference
            events_np = raster.cpu().numpy()
            hidden_np = (self._output_hidden.cpu().numpy()
                         if self._output_hidden is not None
                         else np.zeros_like(events_np))

            onnx_outputs = self._onnx_session.run(
                None, {'events': events_np, 'hidden': hidden_np})

            flow_map_np = onnx_outputs[0].squeeze(0)
            self._output_hidden = torch.tensor(onnx_outputs[1])

            flow_map = torch.tensor(flow_map_np)
            mean_u = flow_map[0].mean().item()
            mean_v = flow_map[1].mean().item()

            # Compute divergence
            centres, vectors = self._quadrant_means(flow_map)
            if self._are_vectors_similar(vectors):
                divergence_val = 0.0
            else:
                ratios = []
                for i, j in self._quadrant_pairs:
                    x1, y1 = centres[i]
                    x2, y2 = centres[j]
                    u1, v1 = vectors[i]
                    u2, v2 = vectors[j]
                    D_pixels = np.hypot(x1 - x2, y1 - y2)
                    D_tips = np.hypot(
                        (x1 + u1) - (x2 + u2), (y1 + v1) - (y2 + v2))
                    ratios.append(self._divergence_ratio(D_pixels, D_tips))
                divergence_val = float(np.mean(ratios))

            # Low-pass filter
            alpha = self.get_parameter(
                'sensor_alpha').get_parameter_value().double_value
            self.filtered_flow_x = (
                alpha * mean_u + (1.0 - alpha) * self.filtered_flow_x)
            self.filtered_flow_y = (
                alpha * mean_v + (1.0 - alpha) * self.filtered_flow_y)

            # Publish
            msg = Float32MultiArray()
            msg.data = [
                self.filtered_flow_x,
                self.filtered_flow_y,
                divergence_val]
            self.publisher_.publish(msg)

            self.get_logger().info(
                f'Flow(x={self.filtered_flow_x:.3f}, '
                f'y={self.filtered_flow_y:.3f}), '
                f'Div={divergence_val:.3f}',
                throttle_duration_sec=1.0)

    def _publish_dummy(self):
        """Publish zero flow data when hardware is unavailable."""
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0]
        self.publisher_.publish(msg)

    @staticmethod
    def _divergence_ratio(D_pixels, D_tips, eps=1e-6):
        return (D_tips - D_pixels) / (D_pixels + eps)

    @staticmethod
    def _are_vectors_similar(vectors, deg_threshold=15.0, min_aligned=3,
                             mag_eps=1e-3):
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

    @staticmethod
    def _quadrant_means(flow):
        _, H, W = flow.shape
        mid_x, mid_y = W // 2, H // 2
        quads = [
            (slice(0, mid_y), slice(0, mid_x)),
            (slice(0, mid_y), slice(mid_x, W)),
            (slice(mid_y, H), slice(mid_x, W)),
            (slice(mid_y, H), slice(0, mid_x)),
        ]
        centres = [
            (W // 4, H // 4), (3 * W // 4, H // 4),
            (3 * W // 4, 3 * H // 4), (W // 4, 3 * H // 4),
        ]
        vectors = []
        for sl_y, sl_x in quads:
            u_mean = flow[0, sl_y, sl_x].mean().item()
            v_mean = flow[1, sl_y, sl_x].mean().item()
            vectors.append((u_mean, v_mean))
        return centres, vectors

    def destroy_node(self):
        self.get_logger().info('Shutting down Speck flow node...')
        self._stop_event.set()
        if self.hardware_available:
            if hasattr(self, '_inference_thread'):
                self._inference_thread.join(timeout=2.0)
            self._readout_filter.stop()
            self._graph.stop()
            self._samna.device.close_device(self._dk)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeckFlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
