import mujoco
import numpy as np
import pygame
import sys
import struct
import math
import time

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import AudioData_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import LaserScan_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__AudioData_
from unitree_sdk2py.idl.default import sensor_msgs_msg_dds__LaserScan_
from unitree_sdk2py.utils.thread import RecurrentThread

import config
if config.ROBOT=="g1":
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
TOPIC_CAMERA_RGB = "rt/camera/rgb"
TOPIC_CAMERA_DEPTH = "rt/camera/depth"
TOPIC_LIDAR_SCAN = "rt/lidar/scan"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35


class KeyboardController:
    """Virtual gamepad controller using keyboard input via pygame.
    Provides the same interface as pygame.joystick.Joystick (get_button, get_axis, get_hat)
    so it can be used as a drop-in replacement.

    Keyboard Mapping:
        Left Stick:   W / A / S / D
        Right Stick:  Arrow Keys (Up/Down/Left/Right)
        D-Pad:        I / J / K / L  (Up/Left/Down/Right)
        A Button:     SPACE
        B Button:     Left Shift
        X Button:     E
        Y Button:     Q
        LB (L1):      Z
        RB (R1):      C
        LT (L2):      1
        RT (R2):      2
        SELECT:       TAB
        START:        ENTER
    """

    KEYMAP_HELP = (
        "\n"
        "  ============= KEYBOARD CONTROLLER =============\n"
        "  Left Stick  : W / A / S / D\n"
        "  Right Stick : Arrow Keys\n"
        "  D-Pad       : I / J / K / L  (Up/Left/Down/Right)\n"
        "  A Button    : SPACE\n"
        "  B Button    : Left Shift\n"
        "  X Button    : E\n"
        "  Y Button    : Q\n"
        "  LB (L1)     : Z\n"
        "  RB (R1)     : C\n"
        "  LT (L2)     : 1\n"
        "  RT (R2)     : 2\n"
        "  SELECT      : TAB\n"
        "  START       : ENTER\n"
        "  ================================================\n"
    )

    def __init__(self):
        # Button mapping: keyboard key -> internal button index
        self._key_to_button = {
            pygame.K_SPACE:  0,   # A
            pygame.K_LSHIFT: 1,   # B
            pygame.K_e:      2,   # X
            pygame.K_q:      3,   # Y
            pygame.K_z:      4,   # LB
            pygame.K_c:      5,   # RB
            pygame.K_TAB:    6,   # SELECT
            pygame.K_RETURN: 7,   # START
        }

        # Axis mapping: (negative_key, positive_key) for each axis index
        self._axis_keys = {
            0: (pygame.K_a, pygame.K_d),         # LX: A(-) / D(+)
            1: (pygame.K_w, pygame.K_s),         # LY: W(-) / S(+)
            3: (pygame.K_LEFT, pygame.K_RIGHT),  # RX: Left(-) / Right(+)
            4: (pygame.K_UP, pygame.K_DOWN),     # RY: Up(-) / Down(+)
        }

        # Trigger keys: axis_index -> key (returns 1.0 when pressed, -1.0 otherwise)
        self._trigger_keys = {
            2: pygame.K_1,  # LT
            5: pygame.K_2,  # RT
        }

        # D-Pad keyboard keys
        self._dpad_keys = {
            "up":    pygame.K_i,
            "down":  pygame.K_k,
            "left":  pygame.K_j,
            "right": pygame.K_l,
        }

        # Internal state
        self._buttons = [0] * 8
        self._axes = [0.0] * 8
        self._hat = (0, 0)

    def init(self):
        """Compatibility stub (pygame.joystick.Joystick has init())."""
        pass

    def update(self):
        """Read current keyboard state and update virtual controller values."""
        keys = pygame.key.get_pressed()

        # --- Buttons ---
        for key, btn_idx in self._key_to_button.items():
            self._buttons[btn_idx] = 1 if keys[key] else 0

        # --- Stick axes (binary: -1 / 0 / +1) ---
        for axis_idx, (neg_key, pos_key) in self._axis_keys.items():
            val = 0.0
            if keys[neg_key]:
                val -= 1.0
            if keys[pos_key]:
                val += 1.0
            self._axes[axis_idx] = val

        # --- Triggers (>0 when pressed) ---
        for axis_idx, key in self._trigger_keys.items():
            self._axes[axis_idx] = 1.0 if keys[key] else -1.0

        # --- D-Pad ---
        dx, dy = 0, 0
        if keys[self._dpad_keys["left"]]:
            dx -= 1
        if keys[self._dpad_keys["right"]]:
            dx += 1
        if keys[self._dpad_keys["down"]]:
            dy -= 1
        if keys[self._dpad_keys["up"]]:
            dy += 1
        self._hat = (max(-1, min(1, dx)), max(-1, min(1, dy)))

    def get_button(self, button_id):
        if 0 <= button_id < len(self._buttons):
            return self._buttons[button_id]
        return 0

    def get_axis(self, axis_id):
        if 0 <= axis_id < len(self._axes):
            return self._axes[axis_id]
        return 0.0

    def get_hat(self, hat_id):
        return self._hat


class UnitreeSdk2Bridge:

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.num_motor = self.mj_model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.have_lidar = False
        self.dt = self.mj_model.opt.timestep
        self.idl_type = (self.num_motor > NUM_MOTOR_IDL_GO) # 0: unitree_go, 1: unitree_hg

        self.joystick = None

        # Discover lidar sensors and cache their sensordata addresses
        self.lidar_sensors = []  # list of (sensor_id, name, sensordata_address)
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name and name.startswith("lidar"):
                addr = self.mj_model.sensor_adr[i]
                self.lidar_sensors.append((i, name, addr))
        self.lidar_sensors.sort(key=lambda x: x[1])  # angular order
        self.num_lidar_rays = len(self.lidar_sensors)
        self._lidar_addrs = [s[2] for s in self.lidar_sensors]

        # Pre-compute LaserScan geometry (computed once)
        if self.num_lidar_rays > 0:
            self.have_lidar = True
            self.lidar_angle_increment = 2.0 * math.pi / self.num_lidar_rays
            self.lidar_angle_min = 0.0
            self.lidar_angle_max = self.lidar_angle_min + self.lidar_angle_increment * (self.num_lidar_rays - 1)
            self.lidar_range_min = 0.05
            self.lidar_range_max = 50000.0
            print(f"Discovered {self.num_lidar_rays} lidar sensors")

        # Check other sensors (IMU, frame)
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == "imu_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

                
        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        )
        self.HighStateThread.Start()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        self.lidar_scan = sensor_msgs_msg_dds__LaserScan_()
        self.lidar_scan_puber = ChannelPublisher(
            TOPIC_LIDAR_SCAN, LaserScan_
        )
        self.lidar_scan_puber.Init()
        self.LidarScanThread = RecurrentThread(
            interval=0.1,
            target=self.PublishLidarScan,
            name="sim_lidar_scan",
        )
        self.LidarScanThread.Start()

        # Setup camera publishers for streaming simulation camera data
        self.setup_camera_publishers()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }

    def LowCmdHandler(self, msg: LowCmd_):
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.mj_data.ctrl[i] = (
                    msg.motor_cmd[i].tau
                    + msg.motor_cmd[i].kp
                    * (msg.motor_cmd[i].q - self.mj_data.sensordata[i])
                    + msg.motor_cmd[i].kd
                    * (
                        msg.motor_cmd[i].dq
                        - self.mj_data.sensordata[i + self.num_motor]
                    )
                )

    def PublishLowState(self):
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
                self.low_state.motor_state[i].dq = self.mj_data.sensordata[
                    i + self.num_motor
                ]
                self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[
                    i + 2 * self.num_motor
                ]

            if self.have_frame_sensor_:

                self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 0
                ]
                self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 1
                ]
                self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 2
                ]
                self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 3
                ]

                self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 4
                ]
                self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 5
                ]
                self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 6
                ]

                self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 7
                ]
                self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 8
                ]
                self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 9
                ]

            if self.joystick != None:
                pygame.event.get()
                if hasattr(self.joystick, 'update'):
                    self.joystick.update()
                # Buttons
                self.low_state.wireless_remote[2] = int(
                    "".join(
                        [
                            f"{key}"
                            for key in [
                                0,
                                0,
                                int(self.joystick.get_axis(self.axis_id["LT"]) > 0),
                                int(self.joystick.get_axis(self.axis_id["RT"]) > 0),
                                int(self.joystick.get_button(self.button_id["SELECT"])),
                                int(self.joystick.get_button(self.button_id["START"])),
                                int(self.joystick.get_button(self.button_id["LB"])),
                                int(self.joystick.get_button(self.button_id["RB"])),
                            ]
                        ]
                    ),
                    2,
                )
                self.low_state.wireless_remote[3] = int(
                    "".join(
                        [
                            f"{key}"
                            for key in [
                                int(self.joystick.get_hat(0)[0] < 0),  # left
                                int(self.joystick.get_hat(0)[1] < 0),  # down
                                int(self.joystick.get_hat(0)[0] > 0), # right
                                int(self.joystick.get_hat(0)[1] > 0),    # up
                                int(self.joystick.get_button(self.button_id["Y"])),     # Y
                                int(self.joystick.get_button(self.button_id["X"])),     # X
                                int(self.joystick.get_button(self.button_id["B"])),     # B
                                int(self.joystick.get_button(self.button_id["A"])),     # A
                            ]
                        ]
                    ),
                    2,
                )
                # Axes
                sticks = [
                    self.joystick.get_axis(self.axis_id["LX"]),
                    self.joystick.get_axis(self.axis_id["RX"]),
                    -self.joystick.get_axis(self.axis_id["RY"]),
                    -self.joystick.get_axis(self.axis_id["LY"]),
                ]
                packs = list(map(lambda x: struct.pack("f", x), sticks))
                self.low_state.wireless_remote[4:8] = packs[0]
                self.low_state.wireless_remote[8:12] = packs[1]
                self.low_state.wireless_remote[12:16] = packs[2]
                self.low_state.wireless_remote[20:24] = packs[3]

            self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):

        if self.mj_data != None:
            self.high_state.position[0] = self.mj_data.sensordata[
                self.dim_motor_sensor + 10
            ]
            self.high_state.position[1] = self.mj_data.sensordata[
                self.dim_motor_sensor + 11
            ]
            self.high_state.position[2] = self.mj_data.sensordata[
                self.dim_motor_sensor + 12
            ]

            self.high_state.velocity[0] = self.mj_data.sensordata[
                self.dim_motor_sensor + 13
            ]
            self.high_state.velocity[1] = self.mj_data.sensordata[
                self.dim_motor_sensor + 14
            ]
            self.high_state.velocity[2] = self.mj_data.sensordata[
                self.dim_motor_sensor + 15
            ]

        self.high_state_puber.Write(self.high_state)

    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            if hasattr(self.joystick, 'update'):
                self.joystick.update()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def PublishLidarScan(self):
        if self.mj_data is None or self.num_lidar_rays == 0:
            return

        # Read ranges from sensordata (negative means no hit)
        raw_ranges = [float(self.mj_data.sensordata[a]) for a in self._lidar_addrs]

        # Convert MuJoCo rangefinder output (-1 = no hit) to inf
        clean_ranges = []
        for d in raw_ranges:
            if d < 0:
                clean_ranges.append(float("inf"))
            else:
                clean_ranges.append(d)

        # Populate LaserScan_ message
        stamp_sec = time.time()
        self.lidar_scan.header.stamp.sec = int(stamp_sec)
        self.lidar_scan.header.stamp.nanosec = int((stamp_sec % 1) * 1e9)
        self.lidar_scan.header.frame_id = "lidar_link"

        self.lidar_scan.angle_min = self.lidar_angle_min
        self.lidar_scan.angle_max = self.lidar_angle_max
        self.lidar_scan.angle_increment = self.lidar_angle_increment
        self.lidar_scan.time_increment = 0.0
        self.lidar_scan.scan_time = float(self.dt)
        self.lidar_scan.range_min = self.lidar_range_min
        self.lidar_scan.range_max = self.lidar_range_max
        self.lidar_scan.ranges = clean_ranges
        self.lidar_scan.intensities = []  # not provided by MuJoCo rangefinder

        self.lidar_scan_puber.Write(self.lidar_scan)
    
    def SetupJoystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
            print(f"Gamepad detected: {self.joystick.get_name()}")
        else:
            print("No gamepad detected — falling back to keyboard controller.")
            self.SetupKeyboard()
            return

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 3,  # Right stick axis x
                "RY": 4,  # Right stick axis y
                "LT": 2,  # Left trigger
                "RT": 5,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def SetupKeyboard(self):
        """Setup a virtual keyboard-based controller (no physical gamepad needed).
        A small pygame window will open — it must stay focused to capture key presses.
        """
        pygame.init()
        # A visible window is required for pygame to capture keyboard events
        screen = pygame.display.set_mode((420, 320))
        pygame.display.set_caption("Unitree Keyboard Controller (keep focused)")

        # Draw a simple help overlay on the window
        screen.fill((30, 30, 30))
        try:
            font = pygame.font.SysFont("monospace", 14)
        except Exception:
            font = pygame.font.Font(None, 16)
        lines = KeyboardController.KEYMAP_HELP.strip().splitlines()
        for i, line in enumerate(lines):
            surf = font.render(line, True, (200, 255, 200))
            screen.blit(surf, (10, 10 + i * 20))
        pygame.display.flip()

        # Create virtual controller
        self.joystick = KeyboardController()
        self.joystick.init()

        # Xbox-style IDs (same convention used by the rest of the code)
        self.axis_id = {
            "LX": 0,
            "LY": 1,
            "RX": 3,
            "RY": 4,
            "LT": 2,
            "RT": 5,
            "DX": 6,
            "DY": 7,
        }
        self.button_id = {
            "X": 2,
            "Y": 3,
            "B": 1,
            "A": 0,
            "LB": 4,
            "RB": 5,
            "SELECT": 6,
            "START": 7,
        }

        print(KeyboardController.KEYMAP_HELP)
        print("Keyboard controller active — keep the pygame window focused.")

    def PrintSceneInformation(self):
        print(" ")

        print("<<------------- Link ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                print("actuator_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name:
                print(
                    "sensor_index:",
                    index,
                    ", name:",
                    name,
                    ", dim:",
                    self.mj_model.sensor_dim[i],
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")

    def setup_camera_publishers(self):
        """Setup camera publishers for streaming simulation camera data"""
        try:
            # Check if cameras are available in the model
            self.available_cameras = []
            for cam_name in ['d435i_rgb', 'd435i_depth']:
                try:
                    mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    self.available_cameras.append(cam_name)
                except:
                    pass
            
            if not self.available_cameras:
                print("⚠️  No cameras found in model - camera streaming disabled")
                return
            
            print(f"✓ Found {len(self.available_cameras)} cameras: {self.available_cameras}")
            
            # Initialize camera publishers using AudioData_ for image streaming
            self.rgb_image = unitree_go_msg_dds__AudioData_()
            self.depth_image = unitree_go_msg_dds__AudioData_()
            
            self.rgb_puber = ChannelPublisher(TOPIC_CAMERA_RGB, AudioData_)
            self.depth_puber = ChannelPublisher(TOPIC_CAMERA_DEPTH, AudioData_)
            
            self.rgb_puber.Init()
            self.depth_puber.Init()
            
            # Camera settings and shared data storage
            self.camera_width, self.camera_height = 320, 240
            self.rendered_rgb_image = None
            self.rendered_depth_image = None
            self.last_camera_render_time = 0
            
            # Get depth camera near/far planes for metric depth conversion
            self.setup_depth_parameters()
            
            # Start camera publishing thread (30 FPS) 
            self.camera_thread = RecurrentThread(
                interval=1.0/30.0,  # 30 FPS
                target=self.PublishCameraData,
                name="sim_camera"
            )
            self.camera_thread.Start()
            
            print("✓ Camera streaming initialized - publishing at 30 FPS")
            print("✓ Camera rendering will be handled by main simulation thread")
            
        except Exception as e:
            print(f"⚠️  Camera setup failed: {e}")
    
    def setup_depth_parameters(self):
        """Get depth camera near/far planes for metric depth conversion"""
        self.depth_near_plane = 0.1
        self.depth_far_plane = 10.0  # Increased range to reduce quantization banding
    
    def setup_camera_renderers_if_needed(self):
        """Setup camera renderers on-demand (called from main simulation thread)"""
        if not hasattr(self, 'rgb_renderer') and hasattr(self, 'available_cameras'):
            try:
                # Create renderers in main simulation thread (safe OpenGL context)
                self.rgb_renderer = mujoco.Renderer(self.mj_model, self.camera_height, self.camera_width)
                self.depth_renderer = mujoco.Renderer(self.mj_model, self.camera_height, self.camera_width)
                self.depth_renderer.enable_depth_rendering()
                print("✓ Camera renderers created in main simulation thread")
            except Exception as e:
                print(f"Camera renderer setup error: {e}")
    
    def render_cameras_for_publishing(self):
        """Render cameras and store in shared variables (called from main simulation thread)"""
        if not hasattr(self, 'available_cameras') or not self.available_cameras:
            return
        
        try:
            # Setup renderers if first time
            self.setup_camera_renderers_if_needed()
            
            if not hasattr(self, 'rgb_renderer'):
                return  # Renderer setup failed
                
            # Render RGB camera
            if 'd435i_rgb' in self.available_cameras:
                self.rgb_renderer.update_scene(self.mj_data, camera='d435i_rgb')
                self.rendered_rgb_image = self.rgb_renderer.render()
            
            # Render depth camera  
            if 'd435i_depth' in self.available_cameras:
                self.depth_renderer.update_scene(self.mj_data, camera='d435i_depth')
                self.rendered_depth_image = self.depth_renderer.render()
            
            # Update timestamp for new data
            import time
            self.last_camera_render_time = time.time()
                
        except Exception as e:
            print(f"Camera rendering error: {e}")
    
    def PublishCameraData(self):
        """Publish RGB + depth camera data via SDK2 using shared rendered images"""
        if not hasattr(self, 'available_cameras') or not self.available_cameras:
            return
        
        # Check if we have new camera data to publish
        if (not hasattr(self, 'last_camera_render_time') or 
            self.rendered_rgb_image is None or 
            self.rendered_depth_image is None):
            return
            
        try:
            import time
            current_time = int(time.time() * 1000)  # milliseconds timestamp
            
            # Publish RGB camera using shared rendered image
            if 'd435i_rgb' in self.available_cameras and self.rendered_rgb_image is not None:
                # Populate AudioData_ message for RGB
                self.rgb_image.time_frame = current_time
                # Convert to list of integers (AudioData_.data is sequence[uint8])
                self.rgb_image.data = self.rendered_rgb_image.flatten().astype(np.uint8).tolist()
                self.rgb_puber.Write(self.rgb_image)
            
            # Publish depth camera using shared rendered image  
            if 'd435i_depth' in self.available_cameras and self.rendered_depth_image is not None:
                depth_image = self.rendered_depth_image
                # Convert depth to single channel if needed
                # if len(depth_image.shape) == 3:
                #     depth_image = depth_image[:,:,0]  # Take first channel
                
                # FIXED: Properly scale MuJoCo metric depth to uint8 range [0,255]
                # MuJoCo returns metric depth in meters, scale to uint8 preserving precision
                # print(f"Depth range: {depth_image.min()} to {depth_image.max()}")
                depth_normalized = (depth_image - self.depth_near_plane) / (self.depth_far_plane - self.depth_near_plane)
                depth_scaled = (depth_normalized * 255).clip(0, 255).astype(np.uint8)
                
                # Populate AudioData_ message for depth
                self.depth_image.time_frame = current_time
                # Convert to list of integers (AudioData_.data is sequence[uint8])
                self.depth_image.data = depth_scaled.flatten().tolist()
                self.depth_puber.Write(self.depth_image)
                
        except Exception as e:
            print(f"Camera publish error: {e}")


class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def PygameKeyUpdate(self):
        """Poll pygame keyboard state for elastic band controls.
        Call this each simulation step when using the keyboard controller.

        Keys:
            7 / Numpad 7  — decrease band length
            8 / Numpad 8  — increase band length
            9 / Numpad 9  — toggle band enable/disable
        """
        if not pygame.get_init():
            return

        keys = pygame.key.get_pressed()

        if keys[pygame.K_7] or keys[pygame.K_KP7]:
            self.length -= 0.01
        if keys[pygame.K_8] or keys[pygame.K_KP8]:
            self.length += 0.01

        # Toggle on key-down edge (avoid repeated toggling while held)
        key9_pressed = keys[pygame.K_9] or keys[pygame.K_KP9]
        if key9_pressed and not getattr(self, '_key9_was_pressed', False):
            self.enable = not self.enable
        self._key9_was_pressed = key9_pressed