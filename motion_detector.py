#!/usr/bin/env python3
"""
Motion Detection CLI Application
Monitors RTSP camera stream for movement and records clips when detected.
Uses YOLO object tracking (ByteTrack) for motion detection.
"""

import argparse
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import platform
import logging
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()


class MotionDetector:
    def __init__(self, rtsp_url, model_size='s', min_confidence=0.30,
                 motion_threshold=75, sustained_frames=3,
                 output_dir="recordings", display=True,
                 enable_fallback=True, fallback_threshold=300,
                 ignore_regions=None):
        """
        Initialize the motion detector with YOLO tracking.

        Args:
            rtsp_url: RTSP stream URL
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            min_confidence: Minimum confidence threshold for detections
            motion_threshold: Minimum pixels movement over sustained_frames to trigger
            sustained_frames: Number of frames to check for sustained movement
            output_dir: Directory to save recordings
            display: Show live video feed
            enable_fallback: Enable fallback motion detection when YOLO fails
            fallback_threshold: Minimum pixel area for fallback motion detection
            ignore_regions: List of regions to ignore for motion detection (x1,y1,x2,y2)
        """
        self.rtsp_url = rtsp_url
        self.model_size = model_size
        self.min_confidence = min_confidence
        self.motion_threshold_param = motion_threshold
        self.sustained_motion_frames_param = sustained_frames
        self.output_dir = Path(output_dir)
        self.display = display
        self.enable_fallback = enable_fallback
        self.fallback_threshold = fallback_threshold
        self.ignore_regions = ignore_regions or []
        self.ignore_mask = None  # Will be created when we get first frame

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize video capture
        self.cap = None
        self.recording = False
        self.manual_recording = False  # Manual recording mode (space bar)
        self.video_writer = None
        self.motion_start_time = None
        self.current_filename = None  # Track current recording filename

        # Create screenshots directory
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)

        # Object tracking: track positions of detected objects
        self.tracked_objects = {}  # {track_id: {'positions': [bbox, ...], 'class': str, 'last_seen': frame_num}}
        self.frame_count = 0
        self.motion_threshold = self.motion_threshold_param
        self.sustained_motion_frames = self.sustained_motion_frames_param
        self.frames_without_motion = 0
        self.motion_cooldown = 60  # Frames to wait before stopping recording (3s at 20fps)

        # Frame validation for corrupted stream handling
        self.expected_frame_size = None
        self.corrupted_frames_count = 0
        self.max_consecutive_corrupted = 10  # Reconnect after 10 consecutive bad frames
        self.frame_size_initialized = False  # Track if we've shown frame size info

        # Fallback motion detection (background subtraction + optical flow)
        self.bg_subtractor = None
        self.prev_frame_gray = None  # For optical flow
        self.fallback_motion_frames = 0  # Track consecutive frames with fallback motion
        self.fallback_sustained_frames = 5  # Require 5 frames of motion before triggering (increased)
        self.fallback_motion_areas = []  # Track motion area sizes for consistency check
        self.bg_learning_frames = 100  # Skip fallback for first 100 frames (longer bg learning)
        self.stable_background_mask = None  # Mask of regions that are part of stable background
        self.background_update_counter = 0  # Counter for background stability tracking

        if self.enable_fallback:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=1000,  # Longer history to better adapt to moving plants
                varThreshold=50,  # Higher threshold to ignore subtle plant motion
                detectShadows=False
            )

        # Live object classes (animals and humans only)
        self.live_object_classes = {
            'person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'bird'
        }

        print(f"Live object classes: {', '.join(sorted(self.live_object_classes))}")

        # Initialize YOLO model with tracking
        print(f"Loading YOLOv8-{model_size} model...")
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
            print(f"YOLOv8-{model_size} model loaded successfully")
            print(f"Minimum confidence: {min_confidence} (filters low-confidence false positives)")
            print(f"Motion threshold: {self.motion_threshold} pixels over {self.sustained_motion_frames} frames")
            if self.enable_fallback:
                print(f"Fallback Detection: Enabled (threshold: {fallback_threshold} pixels)")
        except Exception as e:
            raise RuntimeError(f"Could not load YOLO model: {e}")

    def setup_logging(self):
        """Setup logging with daily rotation"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger('MotionDetector')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Daily rotating file handler
        log_file = logs_dir / "motion_events.log"
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days of logs
            encoding='utf-8'
        )
        file_handler.suffix = "%Y%m%d"  # Add date suffix to rotated logs

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add handler
        self.logger.addHandler(file_handler)

        # Log startup
        self.logger.info("="*60)
        self.logger.info("Motion Detector Started (YOLO Tracking)")
        self.logger.info(f"Model: YOLOv8-{self.model_size}")
        self.logger.info(f"Min Confidence: {self.min_confidence}")
        if self.enable_fallback:
            self.logger.info(f"Fallback Detection: Enabled")
        self.logger.info("="*60)

    def play_alert(self):
        """Play alert sound - uses Mac system sound"""
        print("🔔 Alert!")
        if platform.system() == 'Darwin':  # macOS
            # Use afplay with system sound - play in background
            os.system('afplay /System/Library/Sounds/Glass.aiff &')
        else:
            # Fallback for other systems - terminal bell
            print('\a', flush=True)

    def connect_stream(self, initial=True):
        """Connect to RTSP stream with optimized settings"""
        if initial:
            print(f"Connecting to RTSP stream: {self.rtsp_url}")
        else:
            self.logger.info("Reconnecting to RTSP stream...")

        # Get RTSP transport and buffer size from environment
        rtsp_transport = os.getenv('RTSP_TRANSPORT', 'tcp').lower()
        buffer_size = int(os.getenv('BUFFER_SIZE', '5'))

        # Set RTSP transport protocol (tcp = reliable, udp = low latency)
        # TCP prevents UDP packet loss which causes HEVC "Could not find ref" errors
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'rtsp_transport;{rtsp_transport}'

        # Use FFmpeg backend for better RTSP handling
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # Configure RTSP timeouts
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second connection timeout
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout

        if not self.cap.isOpened():
            raise ConnectionError(f"Failed to connect to RTSP stream: {self.rtsp_url}")

        # Set buffer size to balance latency vs reliability
        # Larger buffer = fewer dropped frames (less HEVC errors) but more latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        if initial:
            print("Successfully connected to stream")
            print(f"Transport: {rtsp_transport.upper()}")
            print(f"Buffer size: {buffer_size} frames")
            print(f"Frame size: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        else:
            self.logger.info(f"Reconnected successfully | Transport: {rtsp_transport.upper()} | Buffer: {buffer_size}")

    def create_ignore_mask(self, frame_shape):
        """Create a mask for ignore regions (e.g., timestamp areas)"""
        height, width = frame_shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255  # Start with all white (include everything)

        for region in self.ignore_regions:
            x1, y1, x2, y2 = region

            # Convert percentages to pixels if needed (values between 0 and 1)
            if 0 < x1 < 1:
                x1 = int(x1 * width)
            if 0 < y1 < 1:
                y1 = int(y1 * height)
            if 0 < x2 < 1:
                x2 = int(x2 * width)
            if 0 < y2 < 1:
                y2 = int(y2 * height)

            # Make sure coordinates are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Black out the ignore region (set to 0)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        return mask

    def is_frame_valid(self, frame):
        """
        Validate frame to detect corruption from RTSP/HEVC errors

        Returns:
            bool: True if frame is valid, False if corrupted
        """
        # Check 1: Frame exists and is not None
        if frame is None:
            return False

        # Check 2: Frame has valid shape
        if len(frame.shape) != 3:
            return False

        height, width, channels = frame.shape

        # Check 3: Frame dimensions are reasonable
        if height < 100 or width < 100 or channels != 3:
            return False

        # Check 4: Store expected size on first valid frame
        if self.expected_frame_size is None:
            self.expected_frame_size = (height, width)
            if not self.frame_size_initialized:
                print(f"Expected frame size set to: {width}x{height}px")
                self.frame_size_initialized = True
            else:
                self.logger.info(f"Frame size reset after reconnection: {width}x{height}px")
            return True

        # Check 5: Frame size matches expected (catches decoder issues)
        if (height, width) != self.expected_frame_size:
            return False

        # Check 6: Check for excessive corruption (gray/black frames)
        # Calculate mean pixel intensity - corrupted frames are often uniform gray
        mean_intensity = np.mean(frame)
        std_intensity = np.std(frame)

        # Valid frames have variation (std > 10)
        # Corrupted frames are often uniform gray (std < 5)
        if std_intensity < 5.0:
            return False

        # Check 7: Look for HEVC artifacts - entire frame very dark or very bright
        if mean_intensity < 10 or mean_intensity > 245:
            return False

        return True

    def detect_motion_and_classify(self, frame):
        """
        Run YOLO tracking on frame and detect motion via position changes.

        Returns:
            tuple: (has_motion, live_objects_list)
        """
        self.frame_count += 1

        # Create ignore mask on first frame
        if self.ignore_mask is None and len(self.ignore_regions) > 0:
            self.ignore_mask = self.create_ignore_mask(frame.shape)
            print(f"Ignore mask created for {len(self.ignore_regions)} region(s)")

        # Run YOLO tracking with enhanced image for nighttime
        # Apply CLAHE for better low-light detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Track objects across frames
        results = self.model.track(
            enhanced_frame,
            conf=self.min_confidence,
            persist=True,  # Keep track IDs across frames
            verbose=False,
            tracker='bytetrack.yaml'  # Use ByteTrack algorithm
        )

        live_objects = []
        has_motion = False

        if results and len(results) > 0:
            boxes = results[0].boxes

            if boxes and boxes.id is not None:
                for box in boxes:
                    # Get detection info
                    track_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    bbox = box.xyxy[0].cpu().numpy()

                    # Only track live objects
                    if class_name not in self.live_object_classes:
                        continue

                    # Calculate bbox center
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    current_center = (center_x, center_y)

                    # Check if object moved with sustained motion requirement
                    if track_id in self.tracked_objects:
                        # Object was seen before - check sustained movement
                        prev_positions = self.tracked_objects[track_id]['positions']

                        # Update position history first
                        self.tracked_objects[track_id]['positions'].append(current_center)
                        if len(self.tracked_objects[track_id]['positions']) > 10:
                            self.tracked_objects[track_id]['positions'].pop(0)
                        self.tracked_objects[track_id]['last_seen'] = self.frame_count

                        # Check if object has moved significantly over last N frames
                        if len(self.tracked_objects[track_id]['positions']) >= self.sustained_motion_frames:
                            # Compare current position to position N frames ago
                            old_position = self.tracked_objects[track_id]['positions'][-(self.sustained_motion_frames)]
                            total_distance = np.sqrt(
                                (current_center[0] - old_position[0]) ** 2 +
                                (current_center[1] - old_position[1]) ** 2
                            )

                            # Require sustained movement over multiple frames
                            if total_distance > self.motion_threshold:
                                has_motion = True
                    else:
                        # New object detected - don't trigger immediately
                        # Wait for sustained movement to avoid jitter on first detection
                        self.tracked_objects[track_id] = {
                            'positions': [current_center],
                            'class': class_name,
                            'last_seen': self.frame_count
                        }

                    # Add to live objects list
                    obj_width = int(bbox[2] - bbox[0])
                    obj_height = int(bbox[3] - bbox[1])
                    live_objects.append({
                        'track_id': track_id,
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'size': (obj_width, obj_height),
                        'area': obj_width * obj_height
                    })

        # Clean up old tracks (not seen for 30 frames)
        tracks_to_remove = []
        for track_id, data in self.tracked_objects.items():
            if self.frame_count - data['last_seen'] > 30:
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]

        # Fallback object detection if YOLO didn't detect anything
        # Detects NEW OBJECTS that appear in scene, not just background movement
        if not has_motion and self.enable_fallback and self.bg_subtractor is not None:
            # Skip fallback during background subtractor learning period
            if self.frame_count <= self.bg_learning_frames:
                return has_motion, live_objects

            # Convert to grayscale for optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame, learningRate=0.001)  # Slow learning to adapt to moving plants

            # Apply ignore mask to remove timestamp/watermark regions
            if self.ignore_mask is not None:
                fg_mask = cv2.bitwise_and(fg_mask, self.ignore_mask)

            # Morphological operations to:
            # 1. Remove noise and small scattered movements (plant leaves)
            # 2. Enhance compact objects (animals)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

            # Remove small noise
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)

            # Close gaps in objects
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)

            # Find contours in the foreground mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze contours to find compact objects (animals) vs scattered patterns (plants)
            valid_objects = []

            for contour in contours:
                area = cv2.contourArea(contour)

                # Minimum size check - cats are at least 1200 pixels (e.g., 40x30)
                if area < 1200:
                    continue

                # Maximum size check - ignore huge areas (likely entire plant moving or lighting change)
                if area > 50000:
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Aspect ratio check - animals are roughly compact (not extremely elongated like branches)
                aspect_ratio = max(w, h) / (min(w, h) + 1)
                if aspect_ratio > 4.0:  # Too elongated = likely not an animal
                    continue

                # Calculate shape metrics to distinguish animals from plants
                # 1. Extent: ratio of contour area to bounding box area
                bbox_area = w * h
                extent = area / bbox_area if bbox_area > 0 else 0

                # 2. Solidity: ratio of contour area to convex hull area
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Animals are relatively compact and solid
                # extent: 0.4-0.9 (compact), plants are scattered: 0.1-0.3
                # solidity: 0.7-0.95 (few indentations), plants have many gaps: 0.3-0.6
                if extent > 0.35 and solidity > 0.65:
                    # This looks like a compact object (potential animal)
                    valid_objects.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'extent': extent,
                        'solidity': solidity,
                        'aspect_ratio': aspect_ratio
                    })

            # If we found valid compact objects, check with optical flow
            has_valid_object = False

            if len(valid_objects) > 0:
                # Check optical flow to confirm real motion (not just noise)
                has_real_flow = False

                if self.prev_frame_gray is not None:
                    # Downsample frames for faster optical flow
                    scale = 0.25
                    small_prev = cv2.resize(self.prev_frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                    # Calculate dense optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        small_prev, small_gray, None,
                        pyr_scale=0.5, levels=2, winsize=10,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                    )

                    # Calculate magnitude of flow
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    # Check if there's significant flow in the object regions
                    for obj in valid_objects:
                        x, y, w, h = obj['bbox']
                        # Scale coordinates to match downsampled flow
                        x_s, y_s = int(x * scale), int(y * scale)
                        w_s, h_s = max(1, int(w * scale)), max(1, int(h * scale))

                        # Extract flow in object region
                        if y_s + h_s <= mag.shape[0] and x_s + w_s <= mag.shape[1]:
                            obj_flow = mag[y_s:y_s+h_s, x_s:x_s+w_s]

                            # Check if object has meaningful motion
                            # Use 75th percentile to detect actual movement
                            if obj_flow.size > 0:
                                flow_75 = np.percentile(obj_flow, 75)
                                if flow_75 > 0.03:  # Threshold for real motion
                                    has_real_flow = True
                                    break
                else:
                    # First frame - accept without flow validation
                    has_real_flow = True

                # Track area consistency for sustained detection
                if has_real_flow:
                    total_area = sum(obj['area'] for obj in valid_objects)
                    self.fallback_motion_areas.append(total_area)
                    if len(self.fallback_motion_areas) > 10:
                        self.fallback_motion_areas.pop(0)

                    # Check area consistency over time
                    area_consistent = True
                    if len(self.fallback_motion_areas) >= 3:
                        areas_array = np.array(self.fallback_motion_areas[-3:])
                        area_ratio = np.max(areas_array) / (np.min(areas_array) + 1)
                        if area_ratio > 6.0:  # Too much variation
                            area_consistent = False

                    if area_consistent:
                        has_valid_object = True

            # Increment or reset sustained detection counter
            if has_valid_object:
                self.fallback_motion_frames += 1
            else:
                self.fallback_motion_frames = 0
                self.fallback_motion_areas.clear()

            # Trigger only after sustained detection over multiple frames
            if self.fallback_motion_frames >= self.fallback_sustained_frames:
                if len(valid_objects) > 0 and len(live_objects) == 0:
                    # Use the largest valid object
                    largest_obj = max(valid_objects, key=lambda o: o['area'])
                    x, y, w, h = largest_obj['bbox']

                    has_motion = True
                    live_objects.append({
                        'track_id': -1,  # Special ID for fallback
                        'class': 'unknown',
                        'confidence': 1.0,
                        'bbox': np.array([x, y, x+w, y+h], dtype=np.float32),
                        'size': (w, h),
                        'area': largest_obj['area']
                    })

            # Store current frame for next iteration
            self.prev_frame_gray = gray

        return has_motion, live_objects

    def start_recording(self, frame, manual=False):
        """Start recording video"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "manual" if manual else "motion"
        filename = self.output_dir / f"{prefix}_{timestamp}.mp4"

        # Use H.264 codec for QuickTime compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        frame_size = (frame.shape[1], frame.shape[0])

        self.video_writer = cv2.VideoWriter(str(filename), fourcc, fps, frame_size)
        self.recording = True
        self.manual_recording = manual
        self.motion_start_time = datetime.now()
        self.current_filename = filename.name  # Store current filename

        print(f"\n{'='*60}")
        if manual:
            print(f"🔴 MANUAL RECORDING STARTED: {filename.name}")
        else:
            print(f"🔴 RECORDING STARTED: {filename.name}")
        print(f"{'='*60}")

        # Log recording start
        if manual:
            self.logger.info(f"MANUAL RECORDING STARTED: {filename.name}")
        else:
            self.logger.info(f"RECORDING STARTED: {filename.name}")

    def stop_recording(self):
        """Stop recording video"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.recording:
            duration = (datetime.now() - self.motion_start_time).total_seconds()
            print(f"{'='*60}")
            if self.manual_recording:
                print(f"⏹️  MANUAL RECORDING STOPPED (Duration: {duration:.1f}s)")
            else:
                print(f"⏹️  RECORDING STOPPED (Duration: {duration:.1f}s)")
            print(f"{'='*60}\n")

            # Log recording stop with filename
            if self.current_filename:
                self.logger.info(f"RECORDING STOPPED: {self.current_filename} | Duration: {duration:.1f}s")
            else:
                self.logger.info(f"RECORDING STOPPED | Duration: {duration:.1f}s")

        self.recording = False
        self.manual_recording = False
        self.motion_start_time = None
        self.current_filename = None

    def save_screenshot(self, frame):
        """Save a screenshot of the current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshots_dir / f"screenshot_{timestamp}.jpg"

        # Save as JPEG with high quality
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"\n📸 Screenshot saved: {filename.name}")
        self.logger.info(f"SCREENSHOT SAVED: {filename.name}")

    def draw_detections(self, frame, objects):
        """Draw detection boxes and labels on frame"""
        display_frame = frame.copy()

        # Draw YOLO tracked objects
        for obj in objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{obj['class']}: {obj['confidence']:.0%} (ID:{obj['track_id']})"

            # Draw box (green)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(display_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add recording indicator
        if self.recording:
            if self.manual_recording:
                cv2.circle(display_frame, (30, 30), 10, (255, 0, 0), -1)  # Blue for manual
                cv2.putText(display_frame, "MANUAL REC", (50, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)  # Red for motion
                cv2.putText(display_frame, "REC", (50, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display_frame

    def run(self):
        """Main detection loop"""
        try:
            self.connect_stream()

            print("\n" + "="*60)
            print("Motion Detection Active (YOLO Tracking)")
            print(f"Model: YOLOv8-{self.model_size}")
            print(f"Min Confidence: {self.min_confidence}")
            print(f"Motion Threshold: {self.motion_threshold} pixels")
            print()
            print("Keyboard Controls:")
            print("  SPACE - Start/Stop manual recording")
            print("  ENTER - Take screenshot")
            print("  Q     - Quit")
            print("="*60 + "\n")

            while True:
                ret, frame = self.cap.read()

                if not ret:
                    self.logger.warning("Failed to read frame, attempting to reconnect...")
                    self.cap.release()
                    self.connect_stream(initial=False)
                    continue

                # Validate frame to filter out RTSP/HEVC corruption
                if not self.is_frame_valid(frame):
                    self.corrupted_frames_count += 1

                    if self.corrupted_frames_count == 1:
                        self.logger.warning("Corrupted frame detected (HEVC/RTSP error) - skipping")

                    # Too many consecutive corrupted frames - reconnect
                    if self.corrupted_frames_count >= self.max_consecutive_corrupted:
                        self.logger.warning(f"Too many corrupted frames ({self.corrupted_frames_count}), reconnecting...")
                        self.cap.release()
                        self.connect_stream(initial=False)
                        self.corrupted_frames_count = 0
                        self.expected_frame_size = None  # Reset frame size
                    continue

                # Frame is valid - reset corruption counter
                if self.corrupted_frames_count > 0:
                    if self.corrupted_frames_count > 1:
                        self.logger.info(f"Stream recovered (skipped {self.corrupted_frames_count} corrupted frames)")
                    self.corrupted_frames_count = 0

                # Detect motion and classify objects
                has_motion, live_objects = self.detect_motion_and_classify(frame)

                # Create annotated frame with detection overlays for recording and display
                annotated_frame = self.draw_detections(frame, live_objects if has_motion else [])

                # Motion detection should only trigger recordings if not in manual recording mode
                if has_motion and len(live_objects) > 0 and not self.manual_recording:
                    self.frames_without_motion = 0

                    # Start recording if not already recording
                    if not self.recording:
                        self.play_alert()
                        self.start_recording(annotated_frame, manual=False)

                        # Print detection info
                        print(f"⚠️  LIVE OBJECT DETECTED!")
                        print(f"   Objects: {len(live_objects)}")

                        # Log motion detection with filename
                        filename_str = f"{self.current_filename} | " if self.current_filename else ""
                        if any(obj['class'] == 'unknown' for obj in live_objects):
                            self.logger.info(f"MOTION DETECTED (FALLBACK): {filename_str}Unknown object moving")
                        else:
                            objects_str = ", ".join([
                                f"{obj['class']}({obj['confidence']:.0%})"
                                for obj in live_objects[:3]  # Log top 3
                            ])
                            self.logger.info(f"MOTION DETECTED: {filename_str}Objects: {objects_str}")

                        print(f"   🐾 LIVE OBJECTS IDENTIFIED:")
                        for i, obj in enumerate(live_objects[:5], 1):  # Show top 5
                            width, height = obj['size']
                            if obj['class'] == 'unknown':
                                print(f"   {i}. UNKNOWN OBJECT (detected via fallback motion)")
                                print(f"      Size: {width}x{height}px")
                            else:
                                confidence_indicator = "✓✓✓" if obj['confidence'] > 0.7 else "✓✓" if obj['confidence'] > 0.4 else "✓"
                                print(f"   {i}. {obj['class'].upper()} {confidence_indicator}")
                                print(f"      Confidence: {obj['confidence']:.1%} | Size: {width}x{height}px | ID: {obj['track_id']}")

                    # Record annotated frame
                    if self.recording and self.video_writer:
                        self.video_writer.write(annotated_frame)
                else:
                    # No motion detected
                    self.frames_without_motion += 1

                    # Continue recording for cooldown period (only for automatic recordings)
                    if self.recording and not self.manual_recording:
                        if self.frames_without_motion < self.motion_cooldown:
                            if self.video_writer:
                                self.video_writer.write(annotated_frame)
                        else:
                            self.stop_recording()

                # Always record if in manual recording mode
                if self.manual_recording and self.recording and self.video_writer:
                    self.video_writer.write(annotated_frame)

                # Display frame
                if self.display:
                    display_frame = annotated_frame

                    # Resize display window to 33% for smaller preview
                    display_height, display_width = display_frame.shape[:2]
                    display_frame_resized = cv2.resize(display_frame,
                                                       (display_width // 3, display_height // 3),
                                                       interpolation=cv2.INTER_LINEAR)

                    cv2.imshow('Motion Detector', display_frame_resized)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    # Q - Quit
                    if key == ord('q'):
                        print("\nShutting down...")
                        break

                    # Space - Toggle manual recording
                    elif key == ord(' '):
                        if self.manual_recording:
                            # Stop manual recording
                            self.stop_recording()
                        else:
                            # Start manual recording
                            if self.recording:
                                # Stop current recording first
                                self.stop_recording()
                            self.start_recording(frame, manual=True)

                    # Enter - Take screenshot
                    elif key == 13 or key == 10:  # Enter key (13 on Windows, 10 on Linux/Mac)
                        self.save_screenshot(frame)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        # Log shutdown
        self.logger.info("Motion Detector Stopped")
        self.logger.info("="*60)

        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description='Motion Detection CLI for RTSP Camera Streams (YOLO Tracking)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (YOLOv8-small)
  python motion_detector.py

  # Use larger model for better accuracy
  python motion_detector.py --model m

  # Use nano model for lower CPU usage
  python motion_detector.py --model n

  # Adjust confidence threshold
  python motion_detector.py --confidence 0.20
        """
    )

    parser.add_argument(
        '--url',
        type=str,
        default=os.getenv('RTSP_URL', 'rtsp://admin:password@192.168.1.108'),
        help='RTSP stream URL (default: from .env file)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='s',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: s)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.30,
        help='Minimum confidence threshold (0.0-1.0, default: 0.30). '
             'Higher = fewer false positives from plants/shadows. '
             'Recommended: 0.25-0.40'
    )

    parser.add_argument(
        '--motion-threshold',
        type=int,
        default=75,
        help='Minimum pixel movement to trigger recording (default: 75). '
             'Lower = more sensitive to movement. Prevents false triggers from bounding box jitter.'
    )

    parser.add_argument(
        '--sustained-frames',
        type=int,
        default=3,
        help='Number of frames to check for sustained movement (default: 3). '
             'Requires consistent movement over multiple frames to filter jitter.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='recordings',
        help='Directory to save recordings (default: recordings)'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable live video display (headless mode)'
    )

    parser.add_argument(
        '--enable-fallback',
        type=lambda x: x.lower() == 'true',
        default=os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true',
        help='Enable fallback motion detection when YOLO fails (default: true)'
    )

    parser.add_argument(
        '--fallback-threshold',
        type=int,
        default=int(os.getenv('FALLBACK_THRESHOLD', '300')),
        help='Minimum pixel area for fallback motion detection (default: 300). '
             'Lower = more sensitive to any movement.'
    )

    parser.add_argument(
        '--ignore-region',
        action='append',
        dest='ignore_regions',
        metavar='X1,Y1,X2,Y2',
        help='Region to ignore for motion detection (e.g., timestamp area). '
             'Format: x1,y1,x2,y2 in pixels or 0.0-1.0 for percentages. '
             'Can be specified multiple times. '
             'Example: --ignore-region 0,0.9,0.3,1.0 (bottom-left 30%% of frame)'
    )

    args = parser.parse_args()

    # Parse ignore regions - check environment first, then CLI args
    ignore_regions = []

    # Default ignore region for timestamp (bottom-left corner - covers full date/time watermark)
    default_regions_str = os.getenv('IGNORE_REGIONS', '0,0.85,0.45,1.0')

    # Parse default/env regions first
    if default_regions_str:
        for region_str in default_regions_str.split(';'):
            region_str = region_str.strip()
            if not region_str:
                continue
            try:
                coords = [float(x.strip()) for x in region_str.split(',')]
                if len(coords) != 4:
                    print(f"Warning: Invalid ignore region '{region_str}'. Must have 4 coordinates. Skipping.")
                    continue
                ignore_regions.append(tuple(coords))
                print(f"Default ignore region added: {coords}")
            except ValueError as e:
                print(f"Warning: Could not parse ignore region '{region_str}': {e}. Skipping.")

    # Add CLI specified regions (these override/add to defaults)
    if args.ignore_regions:
        for region_str in args.ignore_regions:
            try:
                coords = [float(x.strip()) for x in region_str.split(',')]
                if len(coords) != 4:
                    print(f"Warning: Invalid ignore region '{region_str}'. Must have 4 coordinates. Skipping.")
                    continue
                ignore_regions.append(tuple(coords))
                print(f"CLI ignore region added: {coords}")
            except ValueError as e:
                print(f"Warning: Could not parse ignore region '{region_str}': {e}. Skipping.")

    detector = MotionDetector(
        rtsp_url=args.url,
        model_size=args.model,
        min_confidence=args.confidence,
        motion_threshold=args.motion_threshold,
        sustained_frames=args.sustained_frames,
        output_dir=args.output_dir,
        display=not args.no_display,
        enable_fallback=args.enable_fallback,
        fallback_threshold=args.fallback_threshold,
        ignore_regions=ignore_regions
    )

    detector.run()


if __name__ == '__main__':
    main()
