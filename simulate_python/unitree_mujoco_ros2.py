#!/usr/bin/env python3
"""
unitree_mujoco_ros2.py — MuJoCo simulation with ROS2 LaserScan publishing.

Drop-in replacement for unitree_mujoco_cam.py that additionally publishes
LiDAR rangefinder data as sensor_msgs/LaserScan on /scan.

Usage:
    python unitree_mujoco_ros2.py

The script auto-discovers all "lidar*" rangefinder sensors in the loaded
MuJoCo model so it works regardless of ray count or naming convention.
"""

import math
import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


# ───────────────── shared state ─────────────────
locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


# ───────── discover lidar sensors in the model ─────────
def discover_lidar_sensors(model):
    """Find all rangefinder sensors whose name starts with 'lidar'.

    Returns a sorted list of (sensor_id, sensor_name, sensordata_address).
    """
    sensors = []
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name and name.startswith("lidar"):
            addr = model.sensor_adr[i]
            sensors.append((i, name, addr))
    # Sort by name so angular order is preserved (lidar00, lidar01, …)
    sensors.sort(key=lambda x: x[1])
    return sensors


LIDAR_SENSORS = discover_lidar_sensors(mj_model)
NUM_LIDAR_RAYS = len(LIDAR_SENSORS)
print(f"✓ Discovered {NUM_LIDAR_RAYS} lidar sensors in model")

# Pre-compute array of sensordata addresses for fast vectorised reads
_lidar_addrs = [s[2] for s in LIDAR_SENSORS]


# ───────────────── ROS 2 node ─────────────────
class LidarPublisher(Node):
    """Minimal rclpy node that publishes sensor_msgs/LaserScan."""

    def __init__(self):
        super().__init__("mujoco_lidar_publisher")
        self.pub = self.create_publisher(LaserScan, "/scan_test", 10)

        # LaserScan geometry — computed once
        self.num_rays = NUM_LIDAR_RAYS
        if self.num_rays > 0:
            self.angle_increment = 2.0 * math.pi / self.num_rays
        else:
            self.angle_increment = 0.0
        self.angle_min = 0.0
        self.angle_max = self.angle_min + self.angle_increment * (max(self.num_rays - 1, 0))
        self.range_min = 0.05     # metres
        self.range_max = 50000.0  # metres — MuJoCo rays can hit distant ground

        self.get_logger().info(
            f"LaserScan publisher ready: {self.num_rays} rays, "
            f"angle_inc={math.degrees(self.angle_increment):.2f}°, "
            f"publishing on /scan"
        )

    def publish_scan(self, ranges, stamp_sec: float):
        """Build and publish a LaserScan message.

        Args:
            ranges:     list/array of distance values (negative → no hit)
            stamp_sec:  simulation wall-clock time in seconds
        """
        msg = LaserScan()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec % 1) * 1e9)
        msg.header.frame_id = "lidar_link"

        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = float(mj_model.opt.timestep)
        msg.range_min = self.range_min
        msg.range_max = self.range_max

        # Convert MuJoCo rangefinder output (-1 = no hit) → LaserScan (inf = no hit)
        clean = []
        for d in ranges:
            if d < 0:
                clean.append(float("inf"))
            else:
                clean.append(float(d))
        msg.ranges = clean

        self.pub.publish(msg)


# ───────────────── simulation thread ─────────────────
def SimulationThread(lidar_node: LidarPublisher):
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    # Camera rendering throttling
    camera_render_counter = 0
    target_camera_fps = 10.0
    camera_throttle = max(1, int(1.0 / (target_camera_fps * mj_model.opt.timestep)))
    print(f"✓ Camera throttling: rendering every {camera_throttle} sim steps (~{target_camera_fps} FPS)")

    # LiDAR publish throttling (~10 Hz)
    lidar_publish_counter = 0
    target_lidar_fps = 10.0
    lidar_throttle = max(1, int(1.0 / (target_lidar_fps * mj_model.opt.timestep)))
    print(f"✓ LiDAR throttling: publishing every {lidar_throttle} sim steps (~{target_lidar_fps} Hz)")

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            elastic_band.PygameKeyUpdate()
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )

        mujoco.mj_step(mj_model, mj_data)

        # ── read lidar data (while lock is held) ──
        lidar_publish_counter += 1
        if lidar_publish_counter >= lidar_throttle and NUM_LIDAR_RAYS > 0:
            ranges = [float(mj_data.sensordata[a]) for a in _lidar_addrs]
            lidar_node.publish_scan(ranges, time.time())
            lidar_publish_counter = 0

        locker.release()

        # Camera rendering outside lock
        camera_render_counter += 1
        if camera_render_counter >= camera_throttle:
            unitree.render_cameras_for_publishing()
            camera_render_counter = 0

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


# ───────────────── viewer sync thread ─────────────────
def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


# ───────────────── rclpy spin thread ─────────────────
def ROS2SpinThread(node: Node):
    """Spin the rclpy executor in a background thread."""
    try:
        rclpy.spin(node)
    except Exception:
        pass


# ───────────────── main ─────────────────
if __name__ == "__main__":
    rclpy.init()
    lidar_node = LidarPublisher()

    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread, args=(lidar_node,))
    ros2_thread = Thread(target=ROS2SpinThread, args=(lidar_node,), daemon=True)

    viewer_thread.start()
    sim_thread.start()
    ros2_thread.start()

    # Wait for sim to finish (viewer closed)
    sim_thread.join()
    viewer_thread.join()

    # Cleanup
    lidar_node.destroy_node()
    rclpy.shutdown()
