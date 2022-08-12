import os
import numpy as np
import math
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
from carla2gym.carla.PythonAPI.agents.navigation.behavior_agent import BehaviorAgent
from carla2gym.carla.PythonAPI.agents.navigation.controller import VehiclePIDController2, VehiclePIDController3
from carla2gym.carla.util import *
from carla2gym.carla.transform import *
from carla2gym.carla.behavior_planner import BehaviorPlanner
import custom_tasks.envs.configs as carla_configs

import random
import carla
import yaml
import matplotlib.pyplot as plt
f = open(os.path.join(list(carla_configs.__path__)[0], 'carla_navigation.yaml'))
config = yaml.safe_load(f)

class NavigationAgent():
    def __init__(self, env, behavior_planner=False):
        self.target_speed = 30  # km/h
        self.env = env
        self.mycar = env.actor_info['actors'][0]
        self.actor_filers = ['vehicle.*', 'traffic.traffic_light', 'controller.ai.walker', 'walker.*']

        self.vehicle_config = config['vehicle_config']
        self.vehicle_controller = self.SetController()
        self.d_th = config['safety_distance']
        self.t_th = config['safety_time']

        self.collision_distance = config['collision_distance']
        self.behavior_planner = BehaviorPlanner(self.d_th, self.collision_distance, self.t_th) if behavior_planner else False


    def reset(self):
        self.mycar = self.env.actor_info['actors'][0]
        self.vehicle_controller = self.SetController()
        if self.behavior_planner:
            self.behavior_planner.reset(self.env.waypoints)

    def SetController(self):
        dt_longitudinal = config['dt_longitudinal']
        if self.target_speed > 50:
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
        if config['action_dim'] == 2:
            return VehiclePIDController3(self.mycar, args_longitudinal=args_longitudinal_dict,
                                         vehicle_config=self.vehicle_config,
                                         dynamic_lookahead=config['dynamic_lookahead'], lookahead=config['lookahead'])
        elif config['action_dim'] == 3:
            return VehiclePIDController2(self.mycar, args_longitudinal=args_longitudinal_dict, vehicle_config=self.vehicle_config,
                                     dynamic_lookahead=config['dynamic_lookahead'], lookahead=config['lookahead'])

    def step(self, obs):

        # retrieve relevant elements for safe navigation, i.e.: traffic lights and other vehicles
        ego_trans = self.mycar.get_transform()
        ego_loc = ego_trans.location
        ego_yaw = np.radians(ego_trans.rotation.yaw)
        width = self.env.mycar_wpt.lane_width

        vehicle_list = self.env.world.get_actors().filter(self.actor_filers[0])
        vehicle_hazard, target_vehicle = is_vehicle_hazard(self.mycar, self.env.map, vehicle_list, self.d_th)
        if vehicle_hazard:
            return self.get_brake_command()
        if self.env.get_poly_hazard():
            return self.get_brake_command()

        if config['debug_way_preview']:
            showWay(self.env.mycar_wpt, self.env.world, distance=5.0, max_depth=config['max_depth'],
                    lane_change_flag=True, lc_depth=0)

        if config['use_trajectory']:
            matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], [np.sin(ego_yaw), -np.cos(ego_yaw)]])
            local_2_global = lambda x: np.matmul(matrix, x) + np.array([ego_loc.x, ego_loc.y])
            target_loc = carla.Location(*local_2_global(self.env.t_pt), z=ego_loc.z)
            target_vec = carla_vectors_to_delta_vector(target_loc, ego_loc)
            trajectory = get_trajectory(waypoint=self.env.mycar_wpt, target_vec=target_vec,
                                          max_depth=config['max_depth'],
                                          distance=5.0,
                                          lane_change_flag = True,
                                          lc_depth=-1)

            if config['debug_trajectory']:
                draw_trajectory(world=self.env.world, trajectory=trajectory, text_depth=3)

        inter_dist = self.env.inter_dist
        numoflane = self.env.numoflane
        curlane = self.env.curlane
        target_lane = self.env.targetlane
        traffic_light = self.env.traffic_light

        if self.behavior_planner:
            dist_ttc = obs[6:18]
            self.behavior_planner.update_carla_info(ego_loc, self.env.mycar_wpt,
                                                    numoflane, curlane, target_lane,
                                                    inter_dist, traffic_light, dist_ttc)

            behavior = self.behavior_planner.run()
            if config['debug_task']:
                self.behavior_planner.show_behavior(self.env.world, ego_trans)

            if config['debug_ttc']:
                th1 = config['tracking_threshold']
                th2 = config['safety_distance']
                th3 = config['braking_distance']
                # draw roi
                ax = plt.gca()
                coord = []
                if (not self.env.mycar_wpt.is_junction) and behavior[1] == 1:
                    coord.append((width / 2, 0))
                    coord.append((width / 2, 50))
                    coord.append((-width / 2, 50))
                    coord.append((-width / 2, 0))
                    coord.append(coord[0])  # repeat the first point to create a 'closed loop'
                    xs, ys = zip(*coord)  # create lists of x and y values
                    ax.plot(xs, ys)
                coord = []
                for theta in np.arange(0, 2 * np.pi, 0.2):
                    x = th1 * math.cos(theta)
                    y = th1 * math.sin(theta)
                    coord.append([x, y])
                coord.append(coord[0])  # repeat the first point to create a 'closed loop'
                xs, ys = zip(*coord)  # create lists of x and y values
                ax.plot(xs, ys)

                coord = []
                for theta in np.arange(0, 2 * np.pi, 0.2):
                    x = th2 * math.cos(theta)
                    y = th2 * math.sin(theta)
                    coord.append([x, y])
                coord.append(coord[0])  # repeat the first point to create a 'closed loop'
                xs, ys = zip(*coord)  # create lists of x and y values
                ax.plot(xs, ys, 'g')

                coord = []
                for theta in np.arange(0, 2 * np.pi, 0.2):
                    x = th3 * math.cos(theta)
                    y = th3 * math.sin(theta)
                    coord.append([x, y])
                coord.append(coord[0])  # repeat the first point to create a 'closed loop'
                xs, ys = zip(*coord)  # create lists of x and y values
                ax.plot(xs, ys, 'r')
                plt.pause(0.01)

        if config['debug_controller']:
            text = 'Lookahead Distance : {:2f}, Error: {:2f}'.format(self.vehicle_controller._lat_controller.lookahead,
                                                    self.env.lat_dist)
            draw_string(self.env.world, ego_trans, text)

        # speed limit
        self.speed = np.linalg.norm(carla_vector_to_numpy_vector(self.mycar.get_velocity())) * 3.6
        speed_limit = 30 # constant_speed # self.mycar.get_speed_limit()


        # if self.speed_limit <= 30:
        #     self.target_speed = self.speed_limit - 5
        # elif 30 < self.speed_limit < 50:
        #     self.target_speed = self.speed_limit - 8
        # elif self.speed_limit > 50:
        #     self.target_speed = self.speed_limit - 10
        self.target_speed = speed_limit - 5
        self.SetController()

        target_point = self.env.t_pt

        if config['debug_trajectory']:
            left_target_wpt = left_lane_change(self.env.mycar_wpt)
            right_target_wpt = right_lane_change(self.env.mycar_wpt)
            trans_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], [np.sin(ego_yaw), -np.cos(ego_yaw)]])
            global_2_local = lambda x: np.matmul(trans_matrix, x - np.array([ego_loc.x, ego_loc.y]))
            if left_target_wpt:
                self.env.world.debug.draw_point(location=left_target_wpt.transform.location, size=0.5,
                                            color=carla.Color(r=255, g=0, b=255),
                                            life_time=0.01)
                left_t_pt = global_2_local(carla_vector_to_numpy_vector(left_target_wpt.transform.location))
            if right_target_wpt:
                self.env.world.debug.draw_point(location=right_target_wpt.transform.location, size=0.5,
                                                color=carla.Color(r=0, g=255, b=255),
                                                life_time=0.01)
                right_t_pt = global_2_local(carla_vector_to_numpy_vector(right_target_wpt.transform.location))


        if len(self.env.waypoints) == 0:
            return self.get_brake_command()
        else:
            control = self.vehicle_controller.run_step(self.target_speed, target_point)

        # apply control avoiding collision
        if traffic_light != 0:
            control = self.vehicle_controller.run_step(0, target_point)
            control[0] = min(-0.1, control[0])

        if self.behavior_planner:
            if behavior[1] == 1:
                control = self.vehicle_controller.run_step(0, target_point)
                control[0] = min(-0.1, control[0])

        return control

    def get_brake_command(self):
        if config['action_dim'] == 2:
            control = np.array([-1, 0])
        elif config['action_dim'] == 3:
            control = np.array([0, 0, 1])
        return control