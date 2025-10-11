# Motion Detector CLI

A Python CLI application that monitors RTSP camera streams for movement detection, with object classification and automatic video recording.

## Features

- **Anomaly-Based Detection with Continuous Learning**: Learns what's "normal" and adapts to changing weather conditions
  - **Initial Learning**: Analyzes first 200 frames (~10 seconds) to build baseline motion heatmap
  - **Continuous Adaptation**: Heatmap updates every frame to adapt to weather changes (wind picking up/dying down)
  - **Zone-Based Analysis**: Divides scene into 8x8 grid to identify high-motion zones
  - **Traversal Detection**: Distinguishes stationary oscillation from cross-scene movement
- **Live Object Detection**: Only triggers on animals and humans (requires YOLO classification)
  - **Filters plants automatically**: Even large plants that pass anomaly detection are filtered by YOLO
  - **Stationary object filtering**: Identifies static items (plants/trees) that YOLO misclassifies as "person"
    - Tracks bounding box positions over 20+ frames
    - Filters out objects that stay in same location (IoU > 0.6)
    - Prevents false positives from swaying plants/trees
  - **Supported**: person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, bird
  - **Preview shows live objects only**:
    - Green motion boxes: Only shown when live objects are detected
    - Blue YOLO boxes: Only for animals/humans (plants/objects filtered out)
- **Advanced Filtering System**:
  - **Insect Filtering**: Shape-based, brightness, and consecutive frame requirements
  - **Temporal Smoothing**: Running average background to filter brief, rapid movements
- **Object Classification**: Optional YOLO-based object detection to identify what moved (person, cat, dog, etc.)
- **Audio Alerts**: Plays Mac system sound when motion is detected
- **Automatic Recording**: Records video clips in QuickTime-compatible MP4 format
- **Event Logging**: Automatic daily-rotated logs of all motion events with object identification
- **Live Display**: Shows real-time video feed with detection boxes and labels
- **Headless Mode**: Can run without display for server/background operation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your camera URL:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your camera credentials
nano .env
```

Update the RTSP_URL in `.env`:
```bash
RTSP_URL=rtsp://username:password@camera_ip:port/path
```

4. On first run, YOLOv8 will automatically download the model weights (~6MB)

## Usage

### Basic Usage

Run with default settings (uses your camera URL):
```bash
python motion_detector.py
```

### Common Options

**Note:** The system continuously learns and adapts:
- Initial ~10 seconds builds baseline motion patterns
- Then continuously adapts to weather changes (no restart needed!)
- Identifies stationary objects (plants/trees) within 1 second
- Only alerts when YOLO detects moving live objects (animals/humans)

**Detect distant/small cats (more sensitive):**
```bash
python motion_detector.py --min-area 400 --sensitivity 25
```

**Detect only larger animals:**
```bash
python motion_detector.py --min-area 1000 --sensitivity 35
```

**Use custom RTSP URL:**
```bash
python motion_detector.py --url rtsp://user:password@192.168.1.100
```

**Disable object classification for better performance:**
```bash
python motion_detector.py --no-classification
```

**Run in headless mode (no display):**
```bash
python motion_detector.py --no-display
```

**Custom output directory:**
```bash
python motion_detector.py --output-dir /path/to/recordings
```

### All Options

```
--url URL                    RTSP stream URL
                            Default: Loaded from .env file (RTSP_URL)
                            Can be overridden with this flag

--min-area PIXELS           Minimum contour area to detect (pixels)
                            Default: 600
                            Anomaly detection handles plant filtering
                            Recommended values:
                            - 400-600: Very sensitive (distant/small cats)
                            - 600-800: Balanced (default)
                            - 1000+: Large animals only

--sensitivity [1-100]       Motion detection threshold
                            Default: 30
                            Lower = more sensitive
                            Anomaly detection automatically filters plants
                            Recommended: 25-35

--output-dir PATH           Directory for recordings
                            Default: ./recordings

--no-classification         Disable YOLO object classification
                            Improves performance on slower systems

--no-display               Run without video display window
                            Useful for headless servers
```

## How It Works

### Anomaly-Based Motion Detection with Live Object Filtering

Three-layer intelligent detection system:

#### Layer 1: Continuous Learning (All Frames)

1. **Initial Learning** (First 200 frames / ~10 seconds):
   - Builds running average background (alpha=0.05)
   - Divides frame into 8x8 grid and tracks motion frequency per zone
   - Identifies "hot zones" where motion occurs >30% of time
   - No alerts during this phase

2. **Continuous Adaptation** (Every frame after):
   - Zones with motion: frequency increases 2% per frame
   - Zones without motion: frequency decreases 0.4% per frame
   - **Adapts to weather**: Wind picking up/dying down automatically adjusts
   - No restart needed when conditions change!

#### Layer 2: Anomaly Detection (Motion Analysis)

For each motion event, calculates anomaly score:
- **Location Anomaly**: Motion in typically-quiet zones scores higher
  - Zones with <20% historical motion: +2.0 points
  - Zones with <50% historical motion: +0.5 points
- **Traversal Anomaly**: Motion crossing multiple zones scores higher
  - 3+ zones visited: +2.0 points (cat walking)
  - 2 zones visited: +1.0 point

Passes if anomaly score ≥ 2.0 (filters stationary plants in hot zones)

#### Layer 3: Live Object Detection (YOLO Classification + Stationary Filtering)

Final gatekeeper - **requires living things and filters static objects**:
1. YOLO runs on frame with detected motion
2. Checks for animals/humans: `person`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `bird`
3. **Stationary Object Detection** (identifies static items in the scene):
   - Tracks bounding box position of "person" detections over time
   - Calculates IoU (Intersection over Union) to determine if same object
   - If "person" detection stays in same location for 20+ frames → **Marked as stationary**
   - Stationary "person" = plant/tree swaying in wind → **Filtered out**
   - Moving "person" or new detection → **Allowed**
4. **Decision**:
   - ✅ Live object detected AND moving → Start recording
   - ❌ Object is stationary in same location → Reset counter, ignore (static plant/tree)
   - ❌ No live object → Reset counter, ignore

**Result**: Plants/trees that YOLO misclassifies as "person" are identified as stationary objects and filtered!

#### Additional Filters:
- Size filter: Contours below `min-area`
- Insect filters: Shape, brightness, edge proximity, circularity
- Temporal filter: 5 consecutive frames required

### Object Classification

When enabled, YOLOv8 (nano model) runs on frames with detected motion to identify:
- People
- Animals (cat, dog, bird, etc.)
- Vehicles
- 80+ other object classes

### Recording

- Recording starts when live object (animal/human) is detected
- **Continues indefinitely while motion persists** (auto-extends)
- **Minimum 3 seconds after motion stops** (60 frames cooldown)
- Files saved as: `motion_YYYYMMDD_HHMMSS.mp4` (H.264, QuickTime compatible)

**Example scenarios:**
- Cat walks across yard (5 seconds): Records ~8 seconds total (5s movement + 3s cooldown)
- Cat sits and grooms (30 seconds): Records ~33 seconds (30s movement + 3s cooldown)
- Cat pauses mid-walk (2 seconds): Recording continues if motion resumes within 3 seconds

## Examples

### Backyard Cat Monitoring (Default)
```bash
# Intelligent 3-layer detection with continuous learning + stationary filtering
python motion_detector.py  # Uses defaults: --min-area 600 --sensitivity 30

# System automatically:
# - Learns plant patterns in first ~10 seconds
# - Continuously adapts to changing wind/weather
# - Identifies stationary objects (plants/trees) within 1 second
# - Only records when YOLO detects moving animals/humans
# - Filters plants even if misclassified as "person"
```

### High-Sensitivity Mode (Distant/Small Cats)
```bash
# Detect smaller and more distant cats
python motion_detector.py --min-area 400 --sensitivity 25
```

### Large Animals Only
```bash
# Only detect larger animals (reduce CPU load)
python motion_detector.py --min-area 1200 --sensitivity 35
```

### Performance Mode (Not Recommended for Plant Areas)
```bash
# Disable classification for faster processing
python motion_detector.py --no-classification

# WARNING: Disabling classification also disables live object filtering!
# Without YOLO, large plants may trigger false positives
# Only use if CPU is limited AND scene has no problematic plants
```

### Server Deployment
```bash
# Run in background without display
python motion_detector.py --no-display --output-dir /var/recordings
```

## Tuning Sensitivity

### Good News: System Auto-Adapts!
**The system continuously learns and adapts to weather changes:**
- No need to restart when wind conditions change
- Heatmap updates every frame to adjust to new "normal"
- **Stationary object detection**: Automatically identifies static plants/trees within 1 second
- Only moving live objects trigger recording

### Adjustments (Rarely Needed)

If you're **still getting plant false positives**:
- Should be extremely rare with 3-layer detection + stationary filtering
- System automatically learns which "person" detections are static (plants)
- Ensure classification is enabled (default)
- Stationary objects are silently filtered - check logs to verify

If you're getting **insect false positives**:
- **Increase** `--min-area` (e.g., 800, 1000) - most effective for insects
- System has multiple automatic insect filters

If you're **missing real detections** (distant/small cats):
- **Decrease** `--min-area` (e.g., 500, 400)
- **Decrease** `--sensitivity` (e.g., 25, 20)
- Ensure cat appears in zones that don't have frequent plant motion

## Keyboard Controls

While running:
- `q`: Quit the application

## Output Files

### Video Recordings

Recordings are saved in the output directory (default: `./recordings/`) with format:
- Filename: `motion_YYYYMMDD_HHMMSS.mp4`
- Codec: H.264 (mp4v)
- Frame rate: Matches source stream
- Compatible with QuickTime Player

Example: `motion_20251009_143022.mp4`

### Event Logs

All motion events are automatically logged to `logs/motion_events.log` with daily rotation:

**Features:**
- Daily log rotation at midnight
- Rotated logs saved as `motion_events.log.YYYYMMDD`
- Keeps 30 days of history
- Logs include:
  - Application start/stop times
  - Motion detection events
  - Object identification results
  - Recording start/stop with duration

**Example Log Entries:**
```
2025-10-09 22:55:36 | INFO | ============================================================
2025-10-09 22:55:36 | INFO | Motion Detector Started
2025-10-09 22:55:36 | INFO | Min Area: 600 pixels
2025-10-09 22:55:36 | INFO | Sensitivity: 25
2025-10-09 22:55:36 | INFO | Classification: Enabled
2025-10-09 22:55:36 | INFO | ============================================================
2025-10-09 23:12:45 | INFO | MOTION DETECTED | Contours: 2 | Area: 2450px | Frames: 3
2025-10-09 23:12:45 | INFO | OBJECTS DETECTED: cat(85%), potted plant(52%)
2025-10-09 23:12:45 | INFO | RECORDING STARTED: motion_20251009_231245.mp4
2025-10-09 23:12:52 | INFO | RECORDING STOPPED | Duration: 7.2s
```

**Viewing Logs:**
```bash
# View today's logs
cat logs/motion_events.log

# View logs in real-time
tail -f logs/motion_events.log

# Search for specific events
grep "OBJECTS DETECTED" logs/motion_events.log

# View logs from a specific date
cat logs/motion_events.log.20251009
```

## Troubleshooting

### Connection Issues

**"Failed to connect to RTSP stream"**
- Verify the camera URL is correct
- Check network connectivity
- Ensure camera credentials are valid
- Try accessing the stream with VLC player first

### Performance Issues

**Low FPS or lag**
- Disable classification: `--no-classification`
- Reduce camera resolution at the source
- Close unnecessary applications

**High CPU usage**
- Use `--no-classification`
- Increase `--min-area` to reduce false detections
- Use YOLOv8n (nano) model (already default)

### False Detections

**Detecting plants/vegetation in wind**
- **Should be extremely rare** with 3-layer detection + stationary object filtering
- System now identifies static objects:
  - Plants/trees that YOLO misclassifies as "person" are tracked
  - If detected in same location for 20+ frames → filtered as stationary
  - System continuously adapts - no restart needed when weather changes
- If still occurring:
  - Ensure classification is enabled (check "Stationary object filtering: Enabled" on startup)
  - System learns stationary objects automatically within ~1 second
  - Check logs - stationary objects are silently filtered (no recording triggered)

**Detecting insects**
- Increase `--min-area` to 800-1000 (most effective)
- System has multiple automatic insect filters (shape, brightness, edge, consecutive frames)

**Too sensitive to shadows/light changes**
- Increase `--sensitivity` to 35-40
- Ensure camera is not in auto-exposure mode
- Temporal smoothing helps, but extreme light changes may still trigger

**Preview shows boxes around non-living objects**
- This has been fixed - preview should only show animals/humans
- Ensure you're running the latest version
- Non-living objects will NOT trigger recording even if visible in preview briefly

## System Requirements

- **CPU**: Multi-core recommended for object classification
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: Depends on recording duration (approximately 5-10 MB/minute)
- **Network**: Stable connection to camera

## Dependencies

- OpenCV: Video processing and motion detection
- NumPy: Array operations
- Ultralytics (YOLOv8): Object classification
- Pygame: Audio alerts

## License

MIT License - feel free to modify and use as needed.

## Safety & Privacy

This application is designed for legitimate security and monitoring purposes on your own property. Ensure you comply with local privacy laws when recording video.

## Security

- **Never commit the `.env` file** to version control - it contains your camera credentials
- The `.env` file is automatically excluded via `.gitignore`
- Use `.env.example` as a template for others to set up their own configuration
- Consider changing default camera passwords for better security
