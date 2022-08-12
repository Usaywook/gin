import collections

import gym
import numpy as np
import yaml
import cv2
import carla

from carla2gym.carla.PythonAPI.agents.navigation.controller import VehiclePIDController3
from carla2gym.carla.util import left_lane_change, right_lane_change
from carla2gym.carla.transform import carla_vector_to_numpy_vector

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

#     def __init__(self, env):
#         super(NormalizedEnv, self).__init__(env)

    def _action(self, action):
        """restore action"""
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        """normalize action"""
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

    def step(self, action):
        next_state, reward, done, info = self.unwrapped.step(action)
        return next_state, reward, done, info

# Set vehicle control range for Carla environment
class CarlaCtrMask(gym.Wrapper):
    max_throt = 1.0
    max_brake = 1.0
    max_steer = 1.0 # 0.8
    past_steering = 0

    def reset(self):
        # print("CarlaCtrMask: reset")
        self.past_steering = 0
        obs = super().reset()
        return obs

    def step(self, action):
        action_dim = len(action)
        if action_dim == 3:
            throttle, steer, brake = action

        if action_dim == 2:
            acc, steer = action
            # Convert acceleration to throttle and brake
            if acc >= 0.0:
                brake = 0.0
                throttle = min(acc, self.max_throt)
            else:
                throttle = 0.0
                brake = min(abs(acc), self.max_brake)
                # use emergency brake
                # brake = self.max_brake if abs(acc) > 0.1 else min(abs(acc), self.max_brake)

        throttle = np.clip(throttle, 0.0, self.max_throt)
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        brake = np.clip(brake, 0.0, self.max_brake)

        # self.past_steering = steer

        # This a bit biased, but is to avoid fake breaking
        # if brake < 0.1:
        #     brake = 0.0
        # if throttle > brake:
        #     brake = 0.0

        action = np.array([throttle, steer, brake])
        obs, rew, done, info = self.env.step(action)

        return obs, rew, done, info

class CarlaLocalPlannerMask(gym.Wrapper):
    def __init__(self, env):
        super(CarlaLocalPlannerMask, self).__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self._longitudal_speed = 15
        self._preview_distance = 5.0

    def reset(self):
        self.vehicle_controller = self.SetController()
        obs = super().reset()
        return obs

    def step(self, action):

        ego_trans = self.actor_info['actors'][0].get_transform()
        ego_loc = ego_trans.location
        ego_yaw = np.radians(ego_trans.rotation.yaw)

        front_target_wpt = self.mycar_wpt.next(self._preview_distance)[0]
        left_target_wpt = left_lane_change(self.mycar_wpt, waypoint_distance=self._preview_distance)
        right_target_wpt = right_lane_change(self.mycar_wpt, waypoint_distance=self._preview_distance)
        trans_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], [np.sin(ego_yaw), -np.cos(ego_yaw)]])
        global_2_local = lambda x: np.matmul(trans_matrix, x - np.array([ego_loc.x, ego_loc.y]))

        if front_target_wpt:
            # self.world.debug.draw_point(location=front_target_wpt.transform.location, size=0.3,
            #                             color=carla.Color(r=255, g=0, b=0),
            #                             life_time=0.2)
            front_t_pt = global_2_local(carla_vector_to_numpy_vector(front_target_wpt.transform.location))
        else:
            action = 3

        if left_target_wpt:
            # self.world.debug.draw_point(location=left_target_wpt.transform.location, size=0.3,
            #                                 color=carla.Color(r=0, g=255, b=0),
            #                                 life_time=0.2)
            left_t_pt = global_2_local(carla_vector_to_numpy_vector(left_target_wpt.transform.location))
        else:
            left_t_pt = front_t_pt

        if right_target_wpt:
            # self.world.debug.draw_point(location=right_target_wpt.transform.location, size=0.3,
            #                                 color=carla.Color(r=0, g=0, b=255),
            #                                 life_time=0.2)
            right_t_pt = global_2_local(carla_vector_to_numpy_vector(right_target_wpt.transform.location))
        else:
            right_t_pt = front_t_pt


        if action == 0:
            # following
            target_point = front_t_pt
        elif action == 1:
            # left
            target_point = left_t_pt
        elif action == 2:
            # right
            target_point = right_t_pt
        elif action == 3:
            obs, rew, done, info = self.env.step(np.array([-1.0, 0]))
            return obs, rew, done, info
        else:
            raise NotImplementedError

        self.vehicle_controller = self.SetController()
        control = self.vehicle_controller.run_step(self._longitudal_speed, target_point)

        obs, rew, done, info = self.env.step(control)

        return obs, rew, done, info

    def SetController(self, target_speed=20):
        import os
        import configs
        f = open(os.path.join(list(configs.__path__)[0], 'carla_detour.yaml'))
        config = yaml.safe_load(f)
        dt_longitudinal = config['dt_longitudinal']
        if target_speed > 50:
            args_longitudinal_dict = {
                'K_P': config['hw_longitudinal_K_P'],
                'K_D': config['hw_longitudinal_K_D'],
                'K_I': config['hw_longitudinal_K_I'],
                'dt': dt_longitudinal}
        else:
            args_longitudinal_dict = {
                'K_P': config['longitudinal_K_P'],
                'K_D': config['longitudinal_K_D'],
                'K_I': config['longitudinal_K_I'],
                'dt': dt_longitudinal}
        return VehiclePIDController3(self.actor_info['actors'][0], args_longitudinal=args_longitudinal_dict,
                                     vehicle_config=config['vehicle_config'],
                                     dynamic_lookahead=config['dynamic_lookahead'], lookahead=config['lookahead'])

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ActionSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(ActionSkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        total_constraint = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if 'constraint_cost' in info.keys():
                total_constraint += info['constraint_cost']
            total_reward += reward
            if done:
                break
        if 'constraint_cost' in info.keys():
            info['constraint_cost'] = total_constraint
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs