# region imports
# Standard library imports
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline import (
    GStreamerDetectionSimpleApp,
)
from gi.repository import Gst
import hailo
import gi
import socket
import os
import time
from datetime import datetime
import cv2
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

VIDEO_FPS = 25
video_writer_thermal = None
# Third-party imports

gi.require_version("Gst", "1.0")
# Local application-specific imports


hailo_logger = get_logger(__name__)

# endregion imports


# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()


UDP_IP = "127.0.0.1"
UDP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# User-defined callback function: This is the callback function that will be called when data is available from the pipeline


def release_video():
    global video_writer_thermal
    if video_writer_thermal is not None:
        video_writer_thermal.release()
        video_writer_thermal = None
        hailo_logger.info("Video writer released.")


def get_video_writer(width=640, height=480):
    """Create a new VideoWriter with timestamp-based filename."""
    timestamp = datetime.now().strftime("thermal_%y-%m-%d-%H-%M")
    filename = f"{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (width, height))
    hailo_logger.info("Recording to %s", filename)
    return writer


# def app_callback(element, buffer, user_data):
#     frame_idx = user_data.get_count()
#     timestamp = time.time()  # seconds (float)

#     hailo_logger.debug("Processing frame %s", frame_idx)
#     # string_to_print = f"Frame count: {frame_idx} Time: {timestamp}\n"

#     if buffer is None:
#         hailo_logger.warning("Received None buffer at frame=%s", frame_idx)
#         return

#     msgs = []
#     for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(
#         hailo.HAILO_DETECTION
#     ):
#         # string_to_print += (
#         #     f"Detection: {detection.get_label()} Confidence: {detection.get_confidence():.2f}\n"
#         # )
#         bbox = detection.get_bbox()

#         x_center = bbox.xmin() + bbox.width() / 2
#         y_center = bbox.ymin() + bbox.height() / 2
#         width = bbox.width()
#         height = bbox.height()

#         # include timestamp per detection (important!)
#         msgs.append(f"{timestamp} {x_center} {y_center} {width} {height}")

#     # print(string_to_print)

#     if msgs:
#         payload = "\n".join(msgs)
#         sock.sendto(payload.encode(), (UDP_IP, UDP_PORT))

# def app_callback(element, buffer, user_data):
#     global video_writer_thermal

#     frame_idx = user_data.get_count()
#     timestamp = time.time()

#     hailo_logger.debug("Processing frame %s", frame_idx)

#     if buffer is None:
#         hailo_logger.warning("Received None buffer at frame=%s", frame_idx)
#         return

#     # --- Get frame ---
#     pad = element.get_static_pad("src")
#     format, width, height = get_caps_from_pad(pad)

#     if format is None or width is None or height is None:
#         hailo_logger.warning("Could not get caps at frame=%s", frame_idx)
#         return

#     frame = get_numpy_from_buffer(buffer, format, width, height)
#     if frame is None:
#         hailo_logger.warning("Null frame at frame=%s", frame_idx)
#         return

#     # Convert RGB -> BGR for cv2
#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#     # --- Initialize writer on first valid frame ---
#     if video_writer_thermal is None:
#         video_writer_thermal = get_video_writer(width, height)

#     # --- Write frame ---
#     video_writer_thermal.write(frame_bgr)

#     # --- Detection UDP sending (unchanged) ---
#     msgs = []
#     for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(
#         hailo.HAILO_DETECTION
#     ):
#         bbox = detection.get_bbox()
#         x_center = bbox.xmin() + bbox.width() / 2
#         y_center = bbox.ymin() + bbox.height() / 2
#         width_d = bbox.width()
#         height_d = bbox.height()
#         print(f"{timestamp} {x_center} {y_center} {width_d} {height_d}")
#         msgs.append(f"{timestamp} {x_center} {y_center} {width_d} {height_d}")

#     if msgs:
#         payload = "\n".join(msgs)
#         sock.sendto(payload.encode(), (UDP_IP, UDP_PORT))

def app_callback(element, buffer, user_data):
    global video_writer_thermal

    best = None

    frame_idx = user_data.get_count()
    timestamp = time.time()

    hailo_logger.debug("Processing frame %s", frame_idx)

    if buffer is None:
        hailo_logger.warning("Received None buffer at frame=%s", frame_idx)
        return

    # --- Get frame ---
    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    if format is None or width is None or height is None:
        hailo_logger.warning("Could not get caps at frame=%s", frame_idx)
        return

    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        hailo_logger.warning("Null frame at frame=%s", frame_idx)
        return

    # Convert RGB -> BGR for cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- Initialize writer on first valid frame ---
    if video_writer_thermal is None:
        video_writer_thermal = get_video_writer(width, height)

    # --- Write frame ---
    video_writer_thermal.write(frame_bgr)

    # --- Detection UDP sending (unchanged) ---
    msgs = []
    for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(
        hailo.HAILO_DETECTION
    ):
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if best is None or confidence > best["confidence"]:
            best = {
                "confidence": confidence,
                "x1": bbox.xmin(),
                "y1": bbox.ymin(),
                "x2": bbox.xmin() + bbox.width(),
                "y2": bbox.ymin() + bbox.height(),
            }

    if best is not None:
        msg = f"{timestamp};({best['x1']}, {best['y1']}, {best['x2']}, {best['y2']});{best['confidence']}"
        print(msg)
        msgs.append(msg)
    if msgs:
        payload = "\n".join(msgs)
        sock.sendto(payload.encode(), (UDP_IP, UDP_PORT))


def main():
    hailo_logger.info("Starting Detection Simple App.")
    user_data = user_app_callback_class()
    app = GStreamerDetectionSimpleApp(app_callback, user_data)
    try:
        app.run()
    finally:
        release_video()  # always runs, even on Ctrl+C


if __name__ == "__main__":
    main()
