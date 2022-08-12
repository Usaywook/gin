#!/usr/bin/env python

from enum import Enum
from collections import deque
import random
import numpy as np
import carla
from carla2gym.carla.PythonAPI.agents.tools.misc import distance_vehicle, compute_magnitude_angle
from carla2gym.carla.util import *

class DRoutePlanner():
    def __init__(self, vehicle, vehicle_list, buffer_size, destination, max_depth, proximity_threshold=15.0):
        self._vehicle = vehicle
        self._vehicle_list = vehicle_list
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._sampling_radius = 5
        self._min_distance = 4

        self._target_waypoint = None
        self._buffer_size = buffer_size
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._waypoints_queue = deque(maxlen=600)
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0]))

        self._last_traffic_light = None
        self._proximity_threshold = proximity_threshold
        self._max_depth = max_depth

        self._destination = carla.Location(*destination)
        self._compute_waypoints_to_destination()

    def reset(self, traj=None):
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._waypoints_queue = deque(maxlen=600)
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0]))
        self._target_waypoint = None
        self._last_traffic_light = None
        self._compute_waypoints_to_destination(traj)

    def set_destination(self, destination):
        self._destination = destination

    def get_destination(self):
        return self._destination

    def _compute_waypoints_to_destination(self, traj=None):
        # path = get_global_path(self._current_waypoint , self._destination, max_depth=self._max_depth,
        #                        distance=self._sampling_radius,
        #                        lane_change_flag=True,
        #                        threshold=2.0)
        # assert path is True, "no path to destination!"

        if traj:
            path = traj
        else:
            end_waypoint = self._map.get_waypoint(self._destination, lane_type=carla.LaneType.Driving)
            path = get_global_path(self._current_waypoint, end_waypoint,
                                   sampling_radius=self._sampling_radius, debug=False)
        assert path !=[], "no path to destination!"

        for p in path:
            self._waypoints_queue.append(p)

    def run_step(self, hazard=False, light=False):
        waypoints = self._get_waypoints()
        if hazard:
            if light:
                red_light, vehicle_front = self._get_hazard(light=light)
            else:
                vehicle_front = self._get_hazard(light=light)
        # red_light = False
        draw_waypoints(self._world, self._waypoints_queue, color=carla.Color(0, 255, 0))

        if hazard:
            if light:
                return waypoints, red_light, vehicle_front
            else:
                return waypoints, vehicle_front
        else:
            return waypoints

    def get_cur_waypoint(self):
        return self._current_waypoint

    def _get_waypoints(self):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        #   Buffering the waypoints
        while len(self._waypoint_buffer) < self._buffer_size:
            if self._waypoints_queue:
                self._waypoint_buffer.append(
                    self._waypoints_queue.popleft())
            else:
                break

        waypoints = []

        for i, waypoint in enumerate(self._waypoint_buffer):
            waypoints.append(
                [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self._target_waypoint = self._waypoint_buffer[0]

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, waypoint in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index - 1):
                self._waypoint_buffer.popleft()

        return waypoints

    def _get_hazard(self, light=False):
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        if light:
            lights_list = actor_list.filter("*traffic_light*")
            # check for the state of the traffic lights
            light_state = self._is_light_red_us_style(lights_list)

        # check possible obstacles
        vehicle_state = self._is_vehicle_hazard()

        if light:
            return light_state, vehicle_state
        else:
            return vehicle_state

    def _is_vehicle_hazard(self):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
             - bool_flag is True if there is a vehicle ahead blocking us
               and False otherwise
             - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        lane_width = ego_vehicle_waypoint.lane_width

        for target_vehicle in self._vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())

            way_dist = get_waypoint_distance(ego_vehicle_waypoint, target_vehicle_waypoint)
            same_direction = is_same_direction(ego_vehicle_waypoint, target_vehicle_waypoint)
            same_lane = (abs(way_dist) < lane_width / 2) and same_direction
            if not same_lane:
                continue
            loc = target_vehicle.get_location()
            is_front = is_within_distance_ahead(loc, ego_vehicle_location,
                                                self._vehicle.get_transform().rotation.yaw,
                                                self._proximity_threshold)
            if is_front:
                return True

        return False

    def _is_light_red_us_style(self, lights_list):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
             - bool_flag is True if there is a traffic light in RED
               affecting us and False otherwise
             - traffic_light is the object itself or None if there is no
               red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            # It is too late. Do not block the intersection! Keep going!
            return False

        if self._target_waypoint is not None:
            if self._target_waypoint.is_intersection:
                potential_lights = []
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 80.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
                        return True
                else:
                    self._last_traffic_light = None

        return False