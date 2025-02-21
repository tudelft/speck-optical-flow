import io

from PIL import Image
import rerun as rr

from tinycmax.visualizer_utils import event_frame_to_image, flow_map_to_image


class RerunVisualizer:
    """
    Live visualizer using Rerun.
    """

    def __init__(self, app_id, server, web, compression, time_window, blueprint):
        rr.init(app_id)
        rr.serve_web() if web else rr.connect_tcp(server)

        self.compression = compression
        self.counter = 0
        self.time_window = time_window / 1e6
        if blueprint is not None:
            self.blueprint = blueprint
            rr.send_blueprint(self.blueprint, make_active=True)

    def set_counter(self):
        rr.set_time_seconds("time", self.counter * self.time_window)
        self.counter += 1

    def event_frame(self, frame, name="events"):
        image = event_frame_to_image(frame)
        self.log_image(name, image, self.compression)

    def flow_map(self, frame, name="flow"):
        image = flow_map_to_image(frame)
        self.log_image(name, image, self.compression)

    @staticmethod
    def log_image(name, image_ndarray, compression=False):
        # compression: none/false, jpeg, png
        if compression:
            with io.BytesIO() as output:
                Image.fromarray(image_ndarray).save(output, format=compression)
                media_type = f"image/{compression.lower()}"
                rr.log(name, rr.EncodedImage(contents=output.getvalue(), media_type=media_type))
        else:
            rr.log(name, rr.Image(image_ndarray))

    @staticmethod
    def log_scalar(name, scalar):
        rr.log(name, rr.Scalar(scalar))
