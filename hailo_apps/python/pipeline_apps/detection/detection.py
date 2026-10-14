# region imports
# Standard library imports
import math

from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from gi.repository import Gst
import hailo
import cv2
import gi
import os
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
import time
from datetime import datetime
import socket
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports

gi.require_version("Gst", "1.0")

# Local application-specific imports

VIDEO_FPS = 25
video_writer_thermal = None

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42

    def new_function(self):
        return "The meaning of life is: "


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

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tunable thresholds — adjust these without touching logic
# ---------------------------------------------------------------------------
LOCK_CONF   = 0.60   # min confidence to acquire (or re-acquire) a lock
SEARCH_CONF = 0.1   # min confidence accepted while already locked (must match/exceed HEF floor)
MAX_RADIUS  = 300    # max pixel distance from last known center to accept as same drone
                     # scale this if your pipeline resolution differs from the training resolution
_tracker = {
    "locked":      False,
    "last_center": None,   # (x, y) in absolute pixels
}

def _box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def _choose_detection(detections, locked, last_center):
    """
    Mirrors the YOLO script logic exactly.
    detections: list of dicts with keys conf, x1, y1, x2, y2 (absolute pixels)
    Returns (chosen_det | None, new_locked, new_last_center)
    """
    if not detections:
        if locked:
            hailo_logger.debug("Drone not found, last known: %s", last_center)
        return None, locked, last_center

    if not locked:
        # Acquire: pick highest-confidence detection (already >= LOCK_CONF filtered by caller)
        best = max(detections, key=lambda d: d["conf"])
        center = _box_center(best["x1"], best["y1"], best["x2"], best["y2"])
        hailo_logger.info("Locked on drone at %s conf=%.3f", center, best["conf"])
        return best, True, center

    # Already locked --------------------------------------------------------
    # 1. High-conf detection → re-lock (handles reappearance after occlusion)
    high_conf = [d for d in detections if d["conf"] >= LOCK_CONF]
    if high_conf:
        best = max(high_conf, key=lambda d: d["conf"])
        center = _box_center(best["x1"], best["y1"], best["x2"], best["y2"])
        hailo_logger.debug("Re-locked at %s conf=%.3f", center, best["conf"])
        return best, True, center

    # 2. Low-conf but within radius → continue tracking
    nearby = []
    for d in detections:
        c = _box_center(d["x1"], d["y1"], d["x2"], d["y2"])
        dist = _dist(c, last_center)
        if dist <= MAX_RADIUS:
            nearby.append((d, dist, c))

    if nearby:
        closest_det, _, center = min(nearby, key=lambda t: t[1])
        hailo_logger.debug("Tracking at %s conf=%.3f", center, closest_det["conf"])
        return closest_det, True, center

    # 3. Nothing nearby → hold last position
    hailo_logger.debug("Drone lost, holding last_center=%s", last_center)
    return None, True, last_center   # stay locked, don't reset center


def app_callback(element, buffer, user_data):
    global video_writer_thermal

    frame_idx = user_data.get_count()
    timestamp = time.time()

    hailo_logger.debug("Processing frame %s", frame_idx)

    if buffer is None:
        hailo_logger.warning("Received None buffer at frame=%s", frame_idx)
        return

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)
    if format is None or width is None or height is None:
        hailo_logger.warning("Could not get caps at frame=%s", frame_idx)
        return

    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        hailo_logger.warning("Null frame at frame=%s", frame_idx)
        return

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # -----------------------------------------------------------------------
    # Video writer — raw frame, no annotations
    # -----------------------------------------------------------------------
    if video_writer_thermal is None:
        hailo_logger.info("Initializing video writer %dx%d", width, height)
        video_writer_thermal = get_video_writer(width, height)

    video_writer_thermal.write(frame_bgr)

    # -----------------------------------------------------------------------
    # Collect all detections from Hailo, convert to absolute pixels
    # -----------------------------------------------------------------------
    conf_floor = SEARCH_CONF if _tracker["locked"] else LOCK_CONF
    raw_detections = []

    for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
        conf = detection.get_confidence()
        if conf < conf_floor:
            continue

        bbox = detection.get_bbox()
        raw_detections.append({
            "conf": conf,
            "x1":   bbox.xmin() * width,
            "y1":   bbox.ymin() * height,
            "x2":   (bbox.xmin() + bbox.width())  * width,
            "y2":   (bbox.ymin() + bbox.height()) * height,
        })

    # -----------------------------------------------------------------------
    # Run tracker logic
    # -----------------------------------------------------------------------
    chosen, _tracker["locked"], _tracker["last_center"] = _choose_detection(
        raw_detections, _tracker["locked"], _tracker["last_center"]
    )

    # -----------------------------------------------------------------------
    # UDP — only send when we have a live detection (not just a held position)
    # -----------------------------------------------------------------------
    if chosen is not None:
        msg = (
            f"{timestamp}"
            f";({chosen['x1']/width:.4f}, {chosen['y1']/height:.4f},"
            f" {chosen['x2']/width:.4f}, {chosen['y2']/height:.4f})"
            f";{chosen['conf']:.4f}"
        )
        print(msg)
        sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

# def app_callback(element, buffer, user_data):
#     global video_writer_thermal

#     frame_idx = user_data.get_count()
#     timestamp = time.time()

#     hailo_logger.debug("Processing frame %s", frame_idx)

#     if buffer is None:
#         hailo_logger.warning("Received None buffer at frame=%s", frame_idx)
#         return

#     pad = element.get_static_pad("src")
#     format, width, height = get_caps_from_pad(pad)
#     if format is None or width is None or height is None:
#         hailo_logger.warning("Could not get caps at frame=%s", frame_idx)
#         return

#     frame = get_numpy_from_buffer(buffer, format, width, height)
#     if frame is None:
#         hailo_logger.warning("Null frame at frame=%s", frame_idx)
#         return

#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#     # -----------------------------------------------------------------------
#     # Collect all detections from Hailo, convert to absolute pixels
#     # -----------------------------------------------------------------------
#     conf_floor = SEARCH_CONF if _tracker["locked"] else LOCK_CONF
#     raw_detections = []

#     for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
#         conf = detection.get_confidence()
#         if conf < conf_floor:
#             continue   # honour our software threshold on top of the HEF floor

#         bbox = detection.get_bbox()
#         raw_detections.append({
#             "conf": conf,
#             "x1":   bbox.xmin() * width,
#             "y1":   bbox.ymin() * height,
#             "x2":   (bbox.xmin() + bbox.width())  * width,
#             "y2":   (bbox.ymin() + bbox.height()) * height,
#         })

#     # -----------------------------------------------------------------------
#     # Run tracker logic
#     # -----------------------------------------------------------------------
#     chosen, _tracker["locked"], _tracker["last_center"] = _choose_detection(
#         raw_detections, _tracker["locked"], _tracker["last_center"]
#     )

#     # -----------------------------------------------------------------------
#     # Draw — only the chosen (tracked) detection
#     # -----------------------------------------------------------------------
#     if chosen is not None:
#         x1, y1, x2, y2 = int(chosen["x1"]), int(chosen["y1"]), int(chosen["x2"]), int(chosen["y2"])
#         cx, cy = _box_center(x1, y1, x2, y2)

#         cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.circle(frame_bgr, (int(cx), int(cy)), 5, (0, 255, 0), -1)

#         label = f"drone {chosen['conf']:.2f}"
#         lsz, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#         ly = max(y1, lsz[1] + 4)
#         cv2.rectangle(frame_bgr, (x1, ly - lsz[1] - 4), (x1 + lsz[0], ly + baseline), (0, 255, 0), cv2.FILLED)
#         cv2.putText(frame_bgr, label, (x1, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

#     # Draw search radius around last known position
#     if _tracker["locked"] and _tracker["last_center"] is not None:
#         lc = _tracker["last_center"]
#         cv2.circle(frame_bgr, (int(lc[0]), int(lc[1])), MAX_RADIUS, (0, 165, 255), 1)

#     status = "LOCKED" if _tracker["locked"] else "SEARCHING"
#     cv2.putText(frame_bgr, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (0, 255, 0) if _tracker["locked"] else (0, 0, 255), 2)

#     # -----------------------------------------------------------------------
#     # Video writer
#     # -----------------------------------------------------------------------
#     if video_writer_thermal is None:
#         hailo_logger.info("Initializing video writer %dx%d", width, height)
#         video_writer_thermal = get_video_writer(width, height)

#     video_writer_thermal.write(frame_bgr)

#     # -----------------------------------------------------------------------
#     # UDP — only send when we have a live detection (not just a held position)
#     # -----------------------------------------------------------------------
#     if chosen is not None:
#         # Normalise back to 0–1 so the receiver is resolution-independent
#         msg = (
#             f"{timestamp}"
#             f";({chosen['x1']/width:.4f}, {chosen['y1']/height:.4f},"
#             f" {chosen['x2']/width:.4f}, {chosen['y2']/height:.4f})"
#             f";{chosen['conf']:.4f}"
#         )
#         hailo_logger.debug("UDP: %s", msg)
#         sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))


# def app_callback(element, buffer, user_data):
#     global video_writer_thermal

#     best = None

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

#     # --- Draw bounding boxes on frame ---
#     for detection in hailo.get_roi_from_buffer(buffer).get_objects_typed(
#         hailo.HAILO_DETECTION
#     ):
#         bbox = detection.get_bbox()
#         confidence = detection.get_confidence()

#         x1 = int(bbox.xmin() * width)
#         y1 = int(bbox.ymin() * height)
#         x2 = int((bbox.xmin() + bbox.width()) * width)
#         y2 = int((bbox.ymin() + bbox.height()) * height)

#         # Draw rectangle
#         cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Draw label + confidence
#         label = f"drone {confidence:.2f}"
#         label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#         label_y = max(y1, label_size[1] + 4)
#         cv2.rectangle(
#             frame_bgr,
#             (x1, label_y - label_size[1] - 4),
#             (x1 + label_size[0], label_y + baseline),
#             (0, 255, 0),
#             cv2.FILLED,
#         )
#         cv2.putText(
#             frame_bgr, label, (x1, label_y - 2),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
#         )

#         # Track best detection for UDP
#         if best is None or confidence > best["confidence"]:
#             best = {
#                 "confidence": confidence,
#                 "x1": bbox.xmin(),
#                 "y1": bbox.ymin(),
#                 "x2": bbox.xmin() + bbox.width(),
#                 "y2": bbox.ymin() + bbox.height(),
#             }

#     # --- Initialize writer on first valid frame ---
#     if video_writer_thermal is None:
#         print(f"Initializing video writer with width={width}, height={height}")
#         video_writer_thermal = get_video_writer(width, height)

#     # --- Write annotated frame ---
#     video_writer_thermal.write(frame_bgr)

#     # --- UDP sending ---
#     if best is not None:
#         msg = f"{timestamp};({best['x1']}, {best['y1']}, {best['x2']}, {best['y2']});{best['confidence']}"
#         print(msg)
#         sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))


# def app_callback(element, buffer, user_data):
#     global video_writer_thermal

#     best = None

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
#         confidence = detection.get_confidence()
#         if best is None or confidence > best["confidence"]:
#             best = {
#                 "confidence": confidence,
#                 "x1": bbox.xmin(),
#                 "y1": bbox.ymin(),
#                 "x2": bbox.xmin() + bbox.width(),
#                 "y2": bbox.ymin() + bbox.height(),
#             }

#     if best is not None:
#         msg = f"{timestamp};({best['x1']}, {best['y1']}, {best['x2']}, {best['y2']});{best['confidence']}"
#         print(msg)
#         msgs.append(msg)
#     if msgs:
#         payload = "\n".join(msgs)
#         sock.sendto(payload.encode(), (UDP_IP, UDP_PORT))


def main():
    hailo_logger.info("Starting Detection App.")
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    try:
        app.run()
    finally:
        release_video()


if __name__ == "__main__":
    main()