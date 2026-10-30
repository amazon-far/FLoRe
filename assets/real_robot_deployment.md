# 🤖 Real-robot Deployment on the Unitree G1

The low-level controller runs **on the robot**: the G1's onboard Jetson Orin NX runs the F/T bridge and
the controller in a Docker container, and the workstation keeps the high-level policy, which needs its
GPU. SSH tunnels keep every socket on `127.0.0.1` on both machines. The scripts are in
`third_party/holosoma_extensions/force_deploy/`:

| Script | Runs on | Role |
| --- | --- | --- |
| `onboard/g1_tunnel.sh` | workstation | SSH tunnels. Robot → workstation: F/T `:8771`, RecordTap `:8779`, cameras and grippers; workstation → robot: the high-level command stream `:8770` |
| `2_ft_bridge.sh` | robot | both F/T sensors → WebSocket `:8771`, payload-compensated, tared and low-passed at 12 Hz |
| `3_run_policy_vive_force.sh` | robot | the low-level controller at 50 Hz on the robot's DDS; refuses a checkpoint that does not match `MODE` before it touches the robot |
| `onboard/rig.sh` | robot | runs one command in the onboard container (linked as `~/flore_rig/rig.sh`) |
| `onboard/sync_to_g1.sh` | workstation | copies the code and the checkpoint to the robot (`~/flore_rig/home`) |
| `onboard/Dockerfile`, `onboard/setup_envs.sh`, `onboard/g1_internet.sh` | robot, workstation | the image and its environments, and a proxy that gives the robot internet access to build them |

**Hardware.** A Unitree G1 (29 DoF) with the gripper–sensor assembly of the
[data collection interface](../README.md#-robot-free-data-collection-interface) on each wrist (`G1 Adapter.stl`), the
F/T sensors plugged into its Jetson, and one Ethernet cable to the workstation, which sits on the robot's
network (e.g. `192.168.123.224/24`). The commands below reach the Jetson (`unitree@192.168.123.164`) as
`g1`: add a `Host g1` entry with that address to `~/.ssh/config`, copy your key (`ssh-copy-id g1`), and
add `unitree` to the Jetson's `docker` group.

### 1. Set up the robot (once)

On the workstation, from `third_party/holosoma_extensions/force_deploy`:

1. Download the checkpoint into `../logs/WBT-Holosoma/pq9shxs7/`
   ([above](../README.md#-pretrained-low-level-controller)).
2. Put your F/T sensors' configs into `bota_config/` (see below), then copy everything to the robot:
   ```bash
   bash onboard/sync_to_g1.sh
   ```
3. Load the image's base onto the robot and give it internet access (the last command stays open):
   ```bash
   docker pull --platform linux/arm64 ubuntu:24.04
   docker save --platform linux/arm64 ubuntu:24.04 | ssh g1 docker load
   bash onboard/g1_internet.sh
   ```
4. On the robot (`ssh g1`), build the image and the environments:
   ```bash
   cd ~/flore_rig/home/FLoRe/third_party/holosoma_extensions/force_deploy/onboard
   docker build --network host --build-arg http_proxy=http://127.0.0.1:3128 \
       --build-arg https_proxy=http://127.0.0.1:3128 -t flore-rig:latest .
   http_proxy=http://127.0.0.1:3128 https_proxy=http://127.0.0.1:3128 \
       ~/flore_rig/rig.sh bash onboard/setup_envs.sh
   ```

`setup_envs.sh` installs, at the versions of the paper's rig, the F/T bridge's environment
(`force_deploy/.venv-bota`: Python 3.12 and the Bota toolkit) and the controller's runtime
(`~/.holosoma_deps/miniconda3/envs/hssim`: Python 3.11, ONNX Runtime, Pinocchio and the Unitree SDK). Both
live in the container's home, `~/flore_rig/home` on the robot, so they survive new images and new syncs.
Re-run `sync_to_g1.sh` after every code change; it refuses while a rig command is running.

**F/T sensor configs.** `2_ft_bridge.sh` reads the Bota driver config and each tool's payload (mass,
center of mass, inertia) from `force_deploy/bota_config/`. They describe your own sensors and grippers,
so the release ships none:

```
bota_config/
├── bota_driver/bota_binary.json
├── bota_payload_utils/payload_compensator_{left,right}.json
└── rnea_dynamic_compensator/rnea_compensator_{left,right}.json
```

The payload compensators serve `FT_ALGO=bota` (gravity only), the RNEA ones the default `FT_ALGO=rnea`
(gravity and inertia). Each hand finds its sensor by serial number: pass `LEFT_SERIAL=` and
`RIGHT_SERIAL=` (a substring of the sensor's name in `/dev/serial/by-id`) before `bash 2_ft_bridge.sh`.

### 2. Run the low-level controller

Hang the robot on its gantry, turn it on, and put it in damping mode and then development mode (`L2+R2`
on the remote), as in
[holosoma's real-robot guide](https://github.com/Louis-Leee/holosoma/blob/975463c97f7396aba77adb02a3c1cb8ddf44ceb1/src/holosoma_inference/docs/workflows/real-robot-wbt.md).
Then, in one terminal each on the workstation, from `third_party/holosoma_extensions/force_deploy`:

```bash
# workstation: the tunnels (keep it open)
bash onboard/g1_tunnel.sh

# robot: both F/T sensors -> :8771
ssh -t g1 flore_rig/rig.sh bash 2_ft_bridge.sh --pipeline

# robot: the low-level controller
ssh -t g1 flore_rig/rig.sh MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 FORCE_HANDS=both \
    FORCE_ESTOP_N=200 RECORD=1 FORCE_CMD_WIRE=1 bash 3_run_policy_vive_force.sh
```

Write `flore_rig/rig.sh`, not `~/flore_rig/rig.sh`: the workstation's shell would expand `~` to its own
home before ssh sends the command. `--pipeline` adds the 6-axis wrench that the high-level policy
observes to the bridge's stream. The controller's settings:

| Setting | Meaning |
| --- | --- |
| `MODE=5pt-unified-anneal` | the 5-keypoint controller (preset `g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt`), loaded from `logs/WBT-Holosoma/pq9shxs7/`; `ONNX=/abs/path.onnx` loads another export |
| `HOLOSOMA_VIVE_RELATIVE=0` | absolute targets: the keypoints on `:8770` are the hand, torso and foot targets in the pelvis frame, as the high-level policy publishes them |
| `FORCE_CMD_WIRE=1` | executes each hand's force command from `:8770`, zeroing those under 1.5 N; `0`, the default, ignores them |
| `FORCE_HANDS=both` | the hands whose F/T sensor is in the loop: `both`, `left`, `right` or `none` (default `left` with absolute targets) |
| `FORCE_ESTOP_N=200` | hard stop: a measured force above this (N) stops the controller as `B` does (default 40, `0` = off) |
| `RECORD=1` | streams the robot's measured state on `:8779` (RecordTap), which the high-level policy reads back |

The keypoint targets and force commands come from the high-level policy, `scripts/run_hl_serving_5pt.sh`
on the workstation (its guide follows with the high-level release). Until you arm the controller it
holds its ready pose. Drive it from the remote, or from the keyboard in its terminal:

| Remote | Keyboard | Action |
| --- | --- | --- |
| `A` | `]` | start: the robot goes to its ready pose and holds it; the F/T sensors are tared, so keep the grippers in free space |
| `start` | `=` | arm or disarm: the keypoints follow `:8770` (arming is refused while they are far from the robot's pose) |
| `B` | `o` | stop: stiff hold where the robot is, disarmed, force commands zeroed; re-home with `Y` (`i`), then `A` again |
| `R2` | `T` | re-tare both F/T sensors, in free space |
| `L1+R1` | – | kill the controller |

The hard stop stops the policy, not the motors: keep the robot on the gantry and a hand on the hardware
E-stop for first runs. To rehearse a launch without moving the robot, add
`PY=/home/rig/FLoRe/third_party/holosoma_extensions/force_deploy/onboard/py_dryrun.sh` to the command:
every check runs, and the final `run_policy` command is printed instead of executed.

### Without F/T sensors: a pure position controller

The low-level controller also runs **without force sensors**, as a pure position controller that tracks
the five keypoints. Skip the F/T bridge and start the controller with `FORCE_HANDS=none`:

```bash
ssh -t g1 flore_rig/rig.sh MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 FORCE_HANDS=none \
    RECORD=1 FORCE_CMD_WIRE=1 bash 3_run_policy_vive_force.sh
```

Both hands' force readings are then zeroed at the source (no measured force, contact phase, integral or
hard stop), their force commands from `:8770` are dropped, and with `FORCE_CMD_WIRE=1` the keyboard's
force keys are off too. The controller keeps polling for the bridge on `:8771` and logs a reconnect
warning every few seconds, which is harmless.
