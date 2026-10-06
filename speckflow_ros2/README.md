# SpeckFlow ROS2 - Optical Flow Drone Control

Closed-loop drone control using optical flow estimated from the Speck DVS sensor and mtf-01p flow sensor. PX4 offboard mode with progressive steps from position-controlled flight to full flow-based attitude control.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ros2 launch speckflow_control speckflow_launch.py              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ flight_control    │  │ mocap_relay      │  │ speck_flow   │  │
│  │ (state machine)   │  │ (timestamp sync) │  │ (SNN infer.) │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
│           │                      │                    │          │
│     PX4 Topics              /vehicle_           /speck/         │
│  /fmu/in/* /fmu/out/*       motion_capture      optical_flow    │
│           │                      │                    │          │
│  ┌────────▼──────────────────────▼────────────────────▼──────┐  │
│  │                    rosbag2 recorder                        │  │
│  │              ~/Developer/data/speckflow/                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Flight Modes

| Mode | Parameter | Control | Steps |
|------|-----------|---------|-------|
| **waypoint** | `flight_mode: waypoint` | Position (TrajectorySetpoint) | 1, 4 |
| **flow_hover** | `flight_mode: flow_hover` | Attitude: flow PI → roll/pitch, distance PI → thrust | 2, 5 |
| **flow_hover_thrust** | `flight_mode: flow_hover_thrust` | Attitude: flow PI → roll/pitch, divergence PI → thrust | 3, 5 |

## State Machine

```
IDLE ──► TAKEOFF ──► MISSION (waypoint mode) ──► LANDING ──► LANDED
                 ├──► FLOW_HOVER (flow_hover mode)
                 └──► FLOW_HOVER_THRUST (flow_hover_thrust mode)

Any state ──► GEOFENCE_HOLD (on violation, permanent until RC takeover)
```

## Requirements

- Ubuntu 22.04 + ROS 2 Humble (tested on Jetson Orin NX, JetPack 6)
- PX4 with uXRCE-DDS bridge to the companion computer
- `px4_msgs` generated from the **same PX4 firmware tree that runs on the flight controller** (not included in this repo, see below)
- For the Speck sensor (`enable_speck:=true`): `samna`, `sinabs`, `torch`, `onnxruntime`, `numpy`

## Setup

```bash
git clone git@github.com:dfordequan/speckflow_ros2.git ~/Developer/speckflow-ros2
cd ~/Developer/speckflow-ros2

# px4_msgs is not tracked (it is git-ignored). Link it from your PX4 message workspace...
ln -s ~/Developer/homingdrone-ros2/src/px4_msgs src/px4_msgs
# ...or clone it and check out the branch matching your PX4 firmware:
# git clone https://github.com/PX4/px4_msgs.git src/px4_msgs

pip install samna sinabs torch onnxruntime numpy   # only needed for the Speck node
```

The `speck_flow` node loads its SNN weights (`qs95vlk2_minGRU_depth1_model.pt` / `.onnx`) from a
hard-coded `model_path` in [speck_flow_node.py](src/speck_flow/speck_flow/speck_flow_node.py).
These weights are not in the repo, so edit that path on a new machine.

## Quick Start

```bash
cd ~/Developer/speckflow-ros2
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/fastdds_profile.xml
source /opt/ros/humble/setup.bash

# Build (first time: px4_msgs takes ~5 min)
colcon build
source install/setup.bash

# Launch (edit config/params.yaml first to select flight_mode)
ros2 launch speckflow_control speckflow_launch.py

# With Speck sensor
ros2 launch speckflow_control speckflow_launch.py enable_speck:=true
```

## Configuration

All parameters are in `src/speckflow_control/config/params.yaml`. Key parameters:

```yaml
flight_mode: 'waypoint'       # waypoint | flow_hover | flow_hover_thrust
flow_source: 'sensor'         # sensor (mtf-01p) | speck
square_size: 3.0              # meters (waypoint mode)
flight_height: -1.3           # NED: -1.3 = 1.3m (high enough for hover thrust est)
geofence_xy: 3.5              # half-width: 7x7m safety box
geofence_z: -3.0              # NED: -3.0 = 3m ceiling
target_flow_u: 0.0            # flow setpoint X (0 = hover)
target_flow_v: 0.0            # flow setpoint Y (0 = hover)
```

### Sensor vs Speck PI Gains

The speck optical flow signal is **~12x larger amplitude** than mtf-01p for the same physical motion, so separate gains are used automatically based on `flow_source`:

| Parameter | mtf-01p (`sensor`) | Speck (`speck`) |
|-----------|-------------------|-----------------|
| kp_flow | 0.35 | speck_kp_flow: 0.025 |
| ki_flow | 0.17 | speck_ki_flow: 0.01 |

### Speck Flow Preprocessing Pipeline

The speck callback applies these steps in order:

1. **Auto-calibration**: First 200 samples (~2.2s at 90Hz) are averaged to compute bias while drone is stationary on ground
2. **Bias subtraction**: `corrected = raw - auto_bias - manual_offset`
3. **Axis negation**: `speck_negate_x: true` (X axis is inverted relative to body frame)
4. **Low-pass filter**: EMA with `speck_alpha: 0.5` (lower = more filtering)

### Speck Flow Axis Mapping (Body Frame)

Calibrated from position-mode square flight (151457, heading=88 deg):

```
flow_x = -0.381 * v_right + 0.003 * v_forward    (r = -0.703)
flow_y =  0.362 * v_forward + 0.060 * v_right     (r = +0.656)
```

| Speck axis | Body velocity | Controller | Sensitivity |
|-----------|--------------|-----------|-------------|
| flow_x | lateral (right = negative) | `_calc_roll` via `target_flow_u` | 0.38 per m/s |
| flow_y | forward (forward = positive) | `_calc_pitch` via `target_flow_v` | 0.36 per m/s |

### Translation Setpoints

To command forward translation, set `target_flow_v` (positive = forward):

| `target_flow_v` | Estimated forward speed |
|-----------------|------------------------|
| 0.05 | ~0.14 m/s |
| 0.07 | ~0.19 m/s |
| 0.10 | ~0.28 m/s |

To command rightward translation, set `target_flow_u` (negative = right):

| `target_flow_u` | Estimated rightward speed |
|-----------------|--------------------------|
| -0.05 | ~0.13 m/s |
| -0.07 | ~0.18 m/s |
| -0.10 | ~0.26 m/s |

### Runtime Parameter Changes

```bash
# Command forward translation (~0.19 m/s)
ros2 param set /flight_control_node target_flow_v 0.07

# Stop translation
ros2 param set /flight_control_node target_flow_v 0.0

# Adjust speck PI gains live
ros2 param set /flight_control_node speck_kp_flow 0.03
ros2 param set /flight_control_node speck_ki_flow 0.015
```

## Safety

- **Geofence**: 7x7m box + 3m ceiling relative to takeoff position
- On violation: switches from attitude to position hold at current location
- Permanent hold until pilot takes over via RC transmitter
- **Attitude limits**: roll/pitch clamped to ±15 deg
- **Thrust limits**: clamped to [0.12, 0.35]
- **Position clamping**: max 0.25m/tick movement in waypoint mode

### Altitude PI Lessons Learned

- **Must have hover thrust estimate**: Set `flight_height: -1.3` so PX4 has time to compute `HoverThrustEstimate` during takeoff. Without it, the node uses `hover_thrust: 0.25` default, but real hover is ~0.198, causing overshoot and oscillation.
- **Verify DDS bridge**: `HoverThrustEstimate` must be in PX4's `dds_topics.yaml` for XRCE-DDS publishing. If 0 messages in rosbag, check the bridge config.
- **Gentle altitude gains**: `kp_alt: 0.15` is safe. Higher values (0.35+) cause altitude oscillation when hover thrust baseline is wrong.
- **Battery sag**: Long flight sessions cause altitude drop (~1 cm/s). `MC_BAT_SCALE_EN=1` in PX4 helps but doesn't fully fix it. Use fresh batteries.

## Data Logging

Rosbag recorded automatically to `~/Developer/data/speckflow/YYYYMMDD/speckflow_YYYYMMDD-HHMMSS/`

Recorded topics:
- `/fmu/out/vehicle_odometry` - drone position/orientation
- `/fmu/out/vehicle_local_position` - local position estimate
- `/fmu/out/vehicle_optical_flow` - mtf-01p flow sensor
- `/fmu/out/distance_sensor` - altitude (distance to ground)
- `/fmu/out/hover_thrust_estimate` - PX4 hover thrust
- `/fmu/out/vehicle_status` - arm/mode state
- `/vehicle_motion_capture` - mocap data (if available)
- `/speck/optical_flow` - Speck flow estimate (if enabled)
- `/fmu/in/trajectory_setpoint` - our position commands
- `/fmu/in/vehicle_attitude_setpoint` - our attitude commands
- `/fmu/in/offboard_control_mode` - heartbeat flags
- `/speckflow/state` - state machine state

Topics with no publisher will simply have no data in the bag (this is fine).

## Packages

### speckflow_control
Main flight control node. State machine with waypoint navigation, flow-based PI control, and geofence safety. 50Hz heartbeat on separate thread + 50Hz control loop.

### mocap_relay
Relays external motion capture data to PX4 with timestamp synchronization. Configurable input/output topics via parameters.

### speck_flow
Speck2f DVS optical flow estimation using spiking neural networks. Publishes `[flow_x, flow_y, divergence]` on `/speck/optical_flow`. Falls back to publishing zeros if hardware is not connected.

---

## Step-by-Step Test Plan

### Pre-Flight Checklist (All Steps)

1. **PX4 running**: Verify micro XRCE-DDS agent is connected
   ```bash
   ros2 topic list | grep fmu
   # Should show /fmu/in/* and /fmu/out/* topics
   ```

2. **Sensor check**:
   ```bash
   ros2 topic echo /fmu/out/vehicle_odometry --once
   ros2 topic echo /fmu/out/distance_sensor --once
   ros2 topic echo /fmu/out/vehicle_optical_flow --once
   ```

3. **RC transmitter**: Ready for manual takeover (switch to POSCTL or MANUAL)

4. **Parameters**: Verify params.yaml has correct `flight_mode` and `flow_source`

5. **After launch, before arming** (node publishes heartbeat but doesn't arm for 1s):
   ```bash
   ros2 topic hz /fmu/in/offboard_control_mode   # should be ~50 Hz
   ros2 param list /flight_control_node           # verify all params loaded
   ros2 topic echo /speckflow/state               # should show IDLE
   ```

---

### Step 1: Autonomous Square Flight

**Goal**: Fly a 3x3m square at 1m height using position control, then land.

**Config** (`params.yaml`):
```yaml
flight_mode: 'waypoint'
square_size: 3.0
flight_height: -1.0
```

**Launch**:
```bash
ros2 launch speckflow_control speckflow_launch.py
```

**Expected behavior**:
1. `IDLE` → heartbeat + dummy setpoints for 1s, then arm + offboard
2. `TAKEOFF` → climb to 1m at start position
3. `MISSION` → yaw toward WP0, fly to WP0, yaw toward WP1, fly to WP1, ... WP4 (return)
4. `LANDING` → descend to ground
5. `LANDED` → land command sent

**Monitor**:
```bash
ros2 topic echo /speckflow/state    # watch state transitions
```

**Verify** (post-flight):
```bash
ros2 bag info ~/Developer/data/speckflow/YYYYMMDD/speckflow_*
# Check that /fmu/out/vehicle_odometry has data

# Plot position trace (should show square pattern)
ros2 bag play <bag_dir> &
ros2 topic echo /fmu/out/vehicle_odometry --field position
```

**Pass criteria**:
- [ ] Drone arms and takes off to ~1m
- [ ] Flies a recognizable square pattern (check odometry trace)
- [ ] Returns to start position
- [ ] Lands successfully
- [ ] Geofence never triggers (all positions within 3.5m of start)
- [ ] Rosbag contains odometry data

---

### Step 1b: Geofence Test

**Goal**: Verify geofence triggers correctly.

**Config**: Set `square_size: 8.0` (larger than 7m geofence) or `geofence_xy: 1.0` (tiny geofence).

**Expected**: Drone should enter `GEOFENCE_HOLD` during flight and hold position until RC takeover.

**Pass criteria**:
- [ ] State transitions to `GEOFENCE_HOLD`
- [ ] Drone holds position (doesn't continue mission)
- [ ] Pilot can take over via RC

---

### Step 2: Flow Sensor Hover

**Goal**: Hover in place using optical flow (mtf-01p) for horizontal control and distance sensor for altitude.

**Config** (`params.yaml`):
```yaml
flight_mode: 'flow_hover'
flow_source: 'sensor'
target_flow_u: 0.0
target_flow_v: 0.0
target_distance: 1.0
```

**Launch**:
```bash
ros2 launch speckflow_control speckflow_launch.py
```

**Expected behavior**:
1. `IDLE` → `TAKEOFF` (position control to 1m)
2. `TAKEOFF` → `FLOW_HOVER` (switches to attitude control)
3. Drone should hover approximately in place
4. Roll/pitch corrections based on optical flow
5. Thrust corrections based on distance sensor

**Monitor**:
```bash
# Watch flow readings
ros2 topic echo /fmu/out/vehicle_optical_flow --field pixel_flow

# Watch attitude commands
ros2 topic echo /fmu/in/vehicle_attitude_setpoint

# Watch state
ros2 topic echo /speckflow/state
```

**Tuning**:
```bash
# If oscillating: reduce gains
ros2 param set /flight_control_node kp_flow 0.03

# If drifting: increase gains
ros2 param set /flight_control_node kp_flow 0.08
ros2 param set /flight_control_node ki_flow 0.15
```

**Pass criteria**:
- [ ] Transition from position to attitude control is smooth
- [ ] Drone maintains approximately 1m altitude (±0.3m)
- [ ] Drone stays approximately in place (within ~1m drift)
- [ ] Roll/pitch commands stay within ±15 deg
- [ ] Thrust stays within [0.12, 0.30]

---

### Step 2b: Translation Command Test

**Goal**: Command lateral/forward flight via flow rate setpoints.

**During hover** (Step 2 running):
```bash
# Fly slowly to the right
ros2 param set /flight_control_node target_flow_u 0.3

# Stop
ros2 param set /flight_control_node target_flow_u 0.0

# Fly slowly forward
ros2 param set /flight_control_node target_flow_v 0.3

# Stop
ros2 param set /flight_control_node target_flow_v 0.0
```

**Pass criteria**:
- [ ] Drone moves in the commanded direction
- [ ] Drone stops when target set back to 0
- [ ] Height remains stable during translation

---

### Step 3: Distance-Based Thrust Control

**Goal**: Use divergence-based thrust instead of distance sensor.

**Config** (`params.yaml`):
```yaml
flight_mode: 'flow_hover_thrust'
flow_source: 'sensor'
```

**Note**: Initially the divergence signal from the mtf-01p may be noisy. Start with distance-sensor-validated gains from Step 2, then gradually test.

**Pass criteria**:
- [ ] Same horizontal behavior as Step 2
- [ ] Altitude maintained via thrust PI (may need gain tuning)
- [ ] Compare altitude stability with Step 2

---

### Step 4: Data Collection with Speck

**Goal**: Fly square with Speck mounted, record both flow sensor and Speck data.

**Config** (`params.yaml`):
```yaml
flight_mode: 'waypoint'
square_size: 3.0
```

**Launch** (with Speck enabled):
```bash
ros2 launch speckflow_control speckflow_launch.py enable_speck:=true
```

**Verify** (post-flight):
```bash
ros2 bag info <bag_dir>
# Check both topics have data:
#   /fmu/out/vehicle_optical_flow  (mtf-01p)
#   /speck/optical_flow           (Speck)
```

**Analysis**: Compare Speck flow vs mtf-01p flow from the same flight. Plot both time series.

**Pass criteria**:
- [ ] Both `/fmu/out/vehicle_optical_flow` and `/speck/optical_flow` have data in rosbag
- [ ] Speck flow correlates with mtf-01p flow (similar direction/magnitude trends)
- [ ] Square flight completes normally (Speck doesn't interfere)

---

### Step 5: Speck Flow Control

**Goal**: Replace mtf-01p with Speck for horizontal control. Altitude stays on distance sensor PI.

**Config** (`params.yaml`):
```yaml
flight_mode: 'flow_hover'
flow_source: 'speck'
flight_height: -1.3
target_distance: 1.3

# Speck preprocessing (calibrated from flight 151457)
speck_alpha: 0.5                  # LP filter (lower = more filtering)
speck_negate_x: true              # X axis inverted
speck_negate_y: false
speck_calibration_samples: 200    # auto-cal (~2.2s at 90Hz)

# Speck PI gains (signal is ~12x larger than mtf-01p)
speck_kp_flow: 0.025
speck_ki_flow: 0.01

# For hover: both zero. For translation: see axis mapping section
target_flow_u: 0.0
target_flow_v: 0.0               # set to 0.05 for ~0.14 m/s forward
```

**Launch**:
```bash
ros2 launch speckflow_control speckflow_launch.py enable_speck:=true
```

**Note**: The speck node must be running (`enable_speck:=true`). Auto-calibration runs during the first ~2s — drone must be stationary on ground during this time. Separate `speck_kp_flow`/`speck_ki_flow` gains are used automatically when `flow_source: 'speck'`.

**Pass criteria**:
- [x] Drone hovers using Speck flow data (achieved 2024-02-24, flight 114841)
- [ ] Comparable stability to mtf-01p flow (Step 2)
- [ ] Translation commands work with Speck source

---

### Step 5b: Speck Divergence for Altitude

**Goal**: Use Speck divergence for vertical thrust control.

**Config** (`params.yaml`):
```yaml
flight_mode: 'flow_hover_thrust'
flow_source: 'speck'
```

**Pass criteria**:
- [ ] Altitude maintained via Speck divergence signal
- [ ] Combined horizontal (Speck flow) + vertical (Speck divergence) control works

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Drone doesn't arm | `ros2 topic echo /fmu/out/vehicle_status` - check arming_state and pre_flight_checks_pass |
| No offboard mode | Verify heartbeat at 50Hz: `ros2 topic hz /fmu/in/offboard_control_mode` |
| Drift in flow hover | Check flow readings. For mtf-01p: adjust `flow_offset_x/y`. For speck: check auto-cal bias in logs |
| Altitude oscillation | Reduce `kp_alt` (0.15 is safe). Check hover_thrust_estimate has messages in bag. If 0, check DDS bridge |
| Altitude slowly dropping | Battery sag. Use fresh battery. Enable `MC_BAT_SCALE_EN=1` in PX4. Check `hover_thrust_estimate` |
| Oscillation (horizontal) | Reduce `kp_flow`/`ki_flow` (sensor) or `speck_kp_flow`/`speck_ki_flow` (speck) |
| Height instability | Check distance sensor: `ros2 topic echo /fmu/out/distance_sensor`. Adjust `kp_alt/ki_alt` |
| Geofence false trigger | Check start position capture in logs. Increase `geofence_xy` if needed |
| Speck not publishing | Check `ros2 topic echo /speck/optical_flow`. If zeros, hardware not detected |
| hover_thrust_estimate = 0 msgs | Add `HoverThrustEstimate` to PX4 `dds_topics.yaml`, restart XRCE-DDS agent |

## Analysis

Post-flight analysis script generates plots from rosbag data.

```bash
# Source workspace (needed for px4_msgs)
source ~/Developer/speckflow-ros2/install/setup.bash

# Analyse latest bag
python3 ~/Developer/data/analyse/analyse_bag.py

# Analyse specific bag
python3 ~/Developer/data/analyse/analyse_bag.py ~/Developer/data/speckflow/20260221/speckflow_20260221-113504
```

**Generated plots** (saved to `~/Developer/data/analyse/YYYYMMDD/speckflow_YYYYMMDD-HHMMSS/`):
- `xy_trace.png` — bird's eye XY position with setpoints
- `altitude.png` — height + distance sensor over time
- `heading.png` — heading vs yaw setpoint
- `position_xyz.png` — X/Y/Z over time with setpoints
- `optical_flow.png` — mtf-01p flow X/Y + quality (raw + low-pass filtered overlay)
- `attitude_setpoints.png` — roll/pitch/thrust (flow modes only)
- `state_timeline.png` — state machine timeline
- `speck_flow.png` — speck flow X/Y/divergence (raw + filtered, only when speck data present)

## File Reference

```
speckflow-ros2/
├── fastdds_profile.xml
├── README.md
├── src/
│   ├── px4_msgs/                              (symlink → homingdrone-ros2)
│   ├── speckflow_control/
│   │   ├── config/params.yaml                 ← edit this for each step
│   │   ├── launch/speckflow_launch.py
│   │   └── speckflow_control/
│   │       └── flight_control_node.py         ← main control logic
│   ├── mocap_relay/
│   │   └── mocap_relay/mocap_relay_node.py
│   └── speck_flow/
│       └── speck_flow/speck_flow_node.py
```
