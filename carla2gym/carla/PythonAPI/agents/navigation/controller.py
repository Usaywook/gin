#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math

import numpy as np

import carla
from carla2gym.carla.PythonAPI.agents.tools.misc import distance_vehicle, get_speed

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle,
                 args_lateral={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
                 args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        self._vehicle = vehicle

        self._world = self._vehicle.get_world()
        self._lon_controller = PIDLongitudinalController(
            self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(
            self._vehicle, **args_lateral)


    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(waypoint)

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
        :param target_speed: target speed in Km/h
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations
        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), 0.0, 1.0)

class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        if type(waypoint) == carla.Waypoint:
            w_vec = np.array([waypoint.transform.location.x - v_begin.x,
                              waypoint.transform.location.y -
                              v_begin.y, 0.0])
        if type(waypoint) == list:
            w_vec = np.array([waypoint[0] - v_begin.x,
                              waypoint[1] - v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                         (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)

class Purepersuit():
    '''
    PIDLateralController implements lateral control using Pure Persuit.
    '''
    def __init__(self, vehicle, vehicle_config, dynamic_lookahead, lookahead):
        '''
        :param vehicle:
        :param params:
            A : maximum braking accleration
            B : reaction time in abnormal situation
            C : max steering angle [rad]
            D : length of wheel base
            E : sensitivity level over vetical distance between vehicle and ideal path
        '''
        self._vehicle = vehicle
        self.vehicle_config = vehicle_config
        self.dynamic_lookahead = dynamic_lookahead
        self.lookahead = lookahead

    def run_step(self, target_point):
        target_steer = self._pp_control(target_point)
        return target_steer

    def _pp_control(self, target_point):
        '''
        :param target_point: numpy array of local coordinate points
        :return: steering while angle, which range is [-1, 1]
        '''
        # curvature
        x = -target_point[1]
        y = target_point[0]
        curvature = 2 * x / max((x ** 2 + y ** 2), 1e-6)

        # target steer
        target_steer = math.atan2(self.vehicle_config[3] * curvature, 1)
        target_steer = np.clip(target_steer, -self.vehicle_config[2], self.vehicle_config[2]) / self.vehicle_config[2]
        return target_steer

class VehiclePIDController2():
    def __init__(self, vehicle, args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
                 vehicle_config=[1.5, 0.5, 0.6, 2.7, -3], dynamic_lookahead=False, lookahead=5):

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = Purepersuit(self._vehicle, vehicle_config, dynamic_lookahead, lookahead)


    def run_step(self, target_speed, local_plan):

        throttle = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(local_plan)

        return np.array([throttle, steering, 0])

class PIDLongitudinalController2():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """


    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.
            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations
            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class VehiclePIDController3():
    def __init__(self, vehicle, args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
                 vehicle_config=[1.5, 0.5, 0.6, 2.7, -3], dynamic_lookahead=False, lookahead=5):

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._lon_controller = PIDLongitudinalController2(self._vehicle, **args_longitudinal)
        self._lat_controller = Purepersuit(self._vehicle, vehicle_config, dynamic_lookahead, lookahead)


    def run_step(self, target_speed, local_plan):

        acceleration = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(local_plan)
        return np.array([acceleration, steering])