from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import yaml
import os
import math
from collections import deque

import gym
from gym.spaces import Box, Dict
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize

from carla2gym.core.sensors.derived_sensors import CollisionSensor
from carla2gym.core.sensors.derived_sensors import LaneInvasionSensor
from carla2gym.carla.PythonAPI.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from carla2gym.carla.PythonAPI.agents.navigation.global_route_planner import GlobalRoutePlanner
from carla2gym.carla.behavior_planner import BehaviorPlanner
from carla2gym.carla.render import BirdeyeRender
from carla2gym.carla.util import *
from carla2gym.carla.transform import *
from custom_tasks.envs.utils.preprocess import Graph
from custom_tasks.envs import configs as carla_configs

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})



# os.environ["SDL_VIDEODRIVER"] = "dummy"
f = open(os.path.join(list(carla_configs.__path__)[0], 'carla_navigation.yaml'))
config = yaml.safe_load(f)


class CarlaNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args, carla_port=None, tm_port=None):
        self.scenario = "navigation"
        self.client = None
        self.world = None
        carla_port = args.carla_port if carla_port is None else carla_port
        self.tm_port = args.tm_port if tm_port is None else tm_port
        self.use_graph = args.use_graph if args.use_graph is not None else config['use_graph']
        self.use_tracking = args.use_tracking if args.use_tracking is not None else config['use_tracking']
        self.major = args.major
        self.minor = args.minor
        self.penalty = args.penalty

        self.random_route = config['random_route']
        self.start_location = config['start_location']
        self.end_location = config['end_location']

        self.done_epsilon = config['done_epsilon']
        self.spawn_bound = config['spawn_bound']
        self.traffic_rate = args.traffic_rate if args.traffic_rate is not None else config['traffic_rate']
        self.topview = config['topview']

        self.obs_size = config['obs_size']  # camera resolution size = 256
        self.display_size = config['display_size']  # rendering screen size = 256
        self.obs_range = config['obs_range']  # 32
        self.d_behind = config['d_behind']  # 12
        self.lidar_bin = config['lidar_bin']
        self.pixor_size = config['pixor_size']  # 64
        self.max_past_step = config['max_past_step']
        self.display_route = config['display_route']
        self.debug_polygon = config['debug_polygon']

        self.vehicle_config = config['vehicle_config']

        # for traffic light managing
        if config['use_light_change']:
            self.prev_time = time.time()
            self.light_elapse_time = 0
            self.dt = 0
            self.red_light_time = config['light_time']

        if config['action_dim'] == 2:
            self.action_space = Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)
        elif config['action_dim'] == 3:
            self.action_space = Box(np.array([0., -1.0, 0.]), np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        # Observation_space :
        # [speed, lat_dist, delta_yaw, preview_lateral_dis, preview_delta_yaw]
        # left_margin, right_margin, preview_left_margin, preview_right_margin]
        feature = Box(low=0, high=1, shape=(5,), dtype=np.float32)
        if config['use_dict_obs']:
            observation_space_dict = {'feature': feature}
            if config['use_camera_obs']:
                observation_space_dict.update({'camera': Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)})
            if config['use_birdeye_obs']:
                observation_space_dict.update({'birdeye': Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)})
            if config['use_lidar_obs']:
                observation_space_dict.update({'lidar': Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)})
            if self.use_graph:
                # graph feature : x, y, yaw, vx, vy, ax, ay, wx, wy, feasible mask
                self.graph = Graph(num_node = args.max_object,
                                   max_hop = args.max_hop,
                                   num_frame = args.num_frame,
                                   neighbor_dist = args.neighbor_distance,
                                   neighbor_boundary = args.neighbor_boundary,
                                   sigma = args.sigma,
                                   center = args.center,
                                   weighted_graph= args.weighted_graph)

                self.tp_graph = Graph(num_node = args.max_object,
                                   max_hop = args.max_hop,
                                   num_frame = args.num_frame*2,
                                   neighbor_dist = args.neighbor_distance,
                                   neighbor_boundary = args.neighbor_boundary,
                                   sigma = args.sigma,
                                   center = args.num_frame,
                                   weighted_graph= args.weighted_graph)

                observation_space_dict.update({'graph_feature': Box(low=0, high=1, shape=(10, args.num_frame, args.max_object), dtype=np.float32),
                                               'adjacency': Box(low=0, high=1, shape=(args.max_hop + 1, args.max_object, args.max_object), dtype=np.int64)})
            if self.use_tracking:
                observation_space_dict.update({'tracking': Box(low=0, high=1, shape=(config['tracking_number'], 9), dtype=np.float32)})

            self.observation_space = Dict(observation_space_dict)
        else:
            self.observation_space = feature

        # set actor_info
        self.actor_info = {'actors': [], 'blueprints': [], 'transforms': [], 'sensors': []}

        # print("@@@ Initializing carla server ...")
        while self.client is None:
            print(f"Connecting to carla client with ip : {args.carla_ip}, port: {carla_port}, tm_port :{self.tm_port}")
            self.client = carla.Client(args.carla_ip, carla_port)
            self.client.set_timeout(config['client_time_out'])
            # print("Server version: {}".format(self.client.get_server_version()))

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)

        while self.world is None:
            # print("Connecting to the carla world ...")
            self.world = self.client.load_world(config['town'])
            # self.world = self.client.get_world()
            self.async_world()
            self.destroy()
            weather = carla.WeatherParameters(cloudiness=1.0, precipitation=10.0, sun_altitude_angle=70.0)
            self.world.set_weather(weather)
            # print("Carla world made!")
            # print("init: number of actors : {}".format(len(self.world.get_actors())))

        if args.seed > 0:
            self.seed(args.seed)

        # get map from world
        self.map = self.world.get_map()
        self.mycar_wpt = None

        # set global route planner
        dao = GlobalRoutePlannerDAO(self.map, sampling_resolution=config['sampling_resolution'])
        self.grp = GlobalRoutePlanner(dao)
        self.grp.setup(config['use_lane_change_link'])
        if self.random_route:
            self.waypoints = deque(maxlen=config['max_waypoint'])
        else:
            self.waypoints = deque(maxlen=1000)  # if maxlen is short, then out of roi happens continuously


        # initalize renderer
        if (config['use_camera'] or config['use_birdeye'] or config['use_lidar']) and config['pygame_render']:
            pygame.init()
            screen_cnt = [config['use_camera'], config['use_birdeye'], config['use_lidar']].count(True)
            self.screen = pygame.display.set_mode(
                (self.display_size * screen_cnt, self.display_size),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        if config['use_camera']:
            # Camera sensor
            self.camera_sensor = None
            self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
            self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
            self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
            self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
            self.camera_bp.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            self.camera_bp.set_attribute('sensor_tick', '0.02')

        if config['use_birdeye']:
            # birdeye_render
            pixels_per_meter = self.display_size / self.obs_range
            pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
            birdeye_params = {
                'screen_size': [self.display_size, self.display_size],
                'pixels_per_meter': pixels_per_meter,
                'pixels_ahead_vehicle': pixels_ahead_vehicle
            }
            self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

            # Get pixel grid points
            x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T  # elements numbers

        if config['use_lidar']:
            # Lidar sensor
            self.lidar_sensor = None
            self.lidar_data = None
            self.lidar_height = config['lidar_height']
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '32')
            self.lidar_bp.set_attribute('range', '5000')

        # self.behavior_planner = BehaviorPlanner(config['safety_distance'], config['collision_distance'], config['safety_time'])

        # world settings
        # self.async_world()

        # set spawn_points and global route
        self.roi = [-1000, 1000, -1000, 1000]
        vehicle_spawn_points = self.get_random_spawn_points()
        self.set_actor_info(vehicle_spawn_points)
        self.ego_box = [self.actor_info['actors'][0].bounding_box.extent.x,
                        self.actor_info['actors'][0].bounding_box.extent.y]

        self.set_initial_wpt()

        self.set_spectator()

        for actor in self.world.get_actors().filter('traffic.traffic_light'):
            if actor.is_alive:
                actor.set_red_time(config['light_time'])
                actor.set_yellow_time(config['light_time'])
                actor.set_green_time(config['light_time'])

        self.training = True
        self.social_hazard = False
        self.pred_traj = None
        self.pred_id = None
        self.collision_id = []
        self.sync_world()

    def set_actor_info(self, vehicle_spawn_points):
        blueprint_list = []
        transform_list = []
        actor_list = []
        # generate vehicle actors
        for i in range(len(vehicle_spawn_points)):
            # Set the list of blueprints of vehicle actor
            if i == 0:
                globals()['car{}_blueprint'.format(i)] = random.choice(
                    self.world.get_blueprint_library().filter('vehicle.audi.tt'))
                globals()['car{}_blueprint'.format(i)].set_attribute('role_name', 'hero')
            else:
                globals()['car{}_blueprint'.format(i)] = random.choice(
                    self.world.get_blueprint_library().filter('vehicle.tesla.model3'))
            if globals()['car{}_blueprint'.format(i)].has_attribute('color'):
                if i == 0:
                    color = '255,0,0'
                else:
                    color = '150,150,150'
                globals()['car{}_blueprint'.format(i)].set_attribute('color', color)
            if globals()['car{}_blueprint'.format(i)].has_attribute('driver_id'):
                driver_id = random.choice(
                    globals()['car{}_blueprint'.format(i)].get_attribute('driver_id').recommended_values)
                globals()['car{}_blueprint'.format(i)].set_attribute('driver_id', driver_id)
            if globals()['car{}_blueprint'.format(i)].has_attribute('is_invincible'):
                globals()['car{}_blueprint'.format(i)].set_attribute('is_invincible', 'true')

            # Set the list of transforms of each vehicle actor
            globals()['car{}_transform'.format(i)] = vehicle_spawn_points[i]

            # Set the list of actors
            globals()['car{}'.format(i)] = self.world.try_spawn_actor(globals()['car{}_blueprint'.format(i)],
                                                                      globals()['car{}_transform'.format(i)])

            if globals()['car{}'.format(i)] is not None:
                actor_list.append(globals()['car{}'.format(i)])
                blueprint_list.append(globals()['car{}_blueprint'.format(i)])
                transform_list.append(globals()['car{}_transform'.format(i)])

        self.actor_info.clear()
        self.actor_info['actors'] = actor_list
        self.actor_info['blueprints'] = blueprint_list
        self.actor_info['transforms'] = transform_list

        # Set the sensors
        # Attach collision and lane invasion sensor for mycar(i=0)
        collisions_sensor = None
        lane_invasions_sensor = None
        sensor_list = []
        if self.actor_info['actors'][0] is not None:
            collisions_sensor = CollisionSensor(self.actor_info['actors'][0])
            lane_invasions_sensor = LaneInvasionSensor(self.actor_info['actors'][0])
            sensor_list.append(collisions_sensor)
            sensor_list.append(lane_invasions_sensor)
        self.actor_info['sensors'] = sensor_list

        if config['use_camera']:
            # Add camera sensor
            def get_camera_img(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.camera_img = array

            self.camera_sensor = self.world.try_spawn_actor(self.camera_bp,
                                                            self.camera_trans,
                                                            attach_to=self.actor_info['actors'][0])
            self.camera_sensor.listen(lambda data: get_camera_img(data))

        if config['use_polygon']:
            # Get actors polygon list
            self.vehicle_polygons = []
            vehicle_poly_dict = self._get_actor_polygons()
            self.vehicle_polygons.append(vehicle_poly_dict)
            if config['use_birdeye']:
                # Set ego information for render
                self.birdeye_render.set_hero(self.actor_info['actors'][0], self.actor_info['actors'][0].id)

        if config['use_lidar']:
            def get_lidar_data(data):
                self.lidar_data = data

            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp,
                                                       self.lidar_trans,
                                                       attach_to=self.actor_info['actors'][0])
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

    def _get_actor_polygons(self, draw=False):
        """Get the bounding box polygon of actors.

        Args:
            filt: the filter indicating what type of actors we'll look at.

        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
            {actor_id : global_polygon \in R^{4x2}}
        """
        actor_poly_dict = {}
        for actor in self.actor_info['actors']:
            if not actor.is_alive:
                continue
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            # Get length and width with safety margin
            bb = actor.bounding_box
            vel = actor.get_velocity()
            speed = max(min(np.linalg.norm(np.array([vel.x, vel.y, vel.z])), 5.0) / 5.0, 0.01)
            bb.extent.x = bb.extent.x + self.major * speed # longitude safety bound
            bb.extent.y = bb.extent.y + self.minor * speed # latitude safety bound
            self.ego_box = [bb.extent.x, bb.extent.y]

            # Get global bounding box polygon
            poly = np.array([[vertex.x, vertex.y] for vertex in bb.get_world_vertices(trans)])[[0,3,7,4],:]
            actor_poly_dict[actor.id] = poly

            if draw:
                bb.location = trans.location
                if actor.id in self.collision_id:
                    box_color = carla.Color(config['intensity'],0,0)
                else:
                    box_color = carla.Color(0,0,config['intensity'])

                self.world.debug.draw_box(bb, trans.rotation,
                                          life_time=1.1*config['fixed_delta_seconds'],
                                          color=box_color,
                                          thickness=0.2)

                # for i in range(len(poly)):
                #     j = 0 if i==len(poly) - 1 else i + 1
                #     self.world.debug.draw_line(begin=carla.Location(*poly[i]),
                #                                end=carla.Location(*poly[j]),
                #                                life_time=1.1*config['fixed_delta_seconds'],
                #                                thickness=0.2,
                #                                color=box_color)

        return actor_poly_dict

    def get_poly_hazard(self):
        polygons = list(self.vehicle_polygons[-1].values())
        hazard = np.any([self._is_polygons_intersect(polygons[0], poly) for poly in polygons[1:]])
        return hazard

    def _is_polygons_intersect(self, a, b):
        """
        * Helper function to determine whether there is an intersection between the two polygons described
        * by the lists of vertices. Uses the Separating Axis Theorem
        *
        * @param a an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
        * @param b an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
        * @return true if there is any intersection between the 2 polygons, false otherwise
        """

        polygons = [a, b]
        minA, maxA, projected, i, i1, j, minB, maxB = None, None, None, None, None, None, None, None

        for i in range(len(polygons)):

            # for each polygon, look at each edge of the polygon, and determine if it separates
            # the two shapes
            polygon = polygons[i]
            for i1 in range(len(polygon)):

                # grab 2 vertices to create an edge
                i2 = (i1 + 1) % len(polygon)
                p1 = polygon[i1]
                p2 = polygon[i2]

                # find the line perpendicular to this edge
                normal = { 'x': p2[1] - p1[1], 'y': p1[0] - p2[0] }

                minA, maxA = None, None
                # for each vertex in the first shape, project it onto the line perpendicular to the edge
                # and keep track of the min and max of these values
                for j in range(len(a)):
                    projected = normal['x'] * a[j][0] + normal['y'] * a[j][1]
                    if (minA is None) or (projected < minA):
                        minA = projected

                    if (maxA is None) or (projected > maxA):
                        maxA = projected

                # for each vertex in the second shape, project it onto the line perpendicular to the edge
                # and keep track of the min and max of these values
                minB, maxB = None, None
                for j in range(len(b)):
                    projected = normal['x'] * b[j][0] + normal['y'] * b[j][1]
                    if (minB is None) or (projected < minB):
                        minB = projected

                    if (maxB is None) or (projected > maxB):
                        maxB = projected

                # if there is no overlap between the projects, the edge we are looking at separates the two
                # polygons, and we know there is no overlap
                if (maxA < minB) or (maxB < minA):
                    # print("polygons don't intersect!")
                    return False

            return True

    def _rgb_to_display_surface(self, rgb):
        """ Generate pygame surface given an rgb image uint8 matrix
        """
        surface = pygame.Surface((self.display_size, self.display_size)).convert()
        x = resize(rgb, (self.display_size, self.display_size))
        x = np.flip(x, axis=1)
        x = np.rot90(x, 1)
        pygame.surfarray.blit_array(surface, x)
        return surface

    def _display_to_rgb(self, display):
        """ Transform image grabbed from pygame display to an rgb image uint8 matrix
        """
        rgb = np.fliplr(np.rot90(display, 3))  # flip to regular view
        rgb = resize(rgb, (self.obs_size, self.obs_size))  # resize
        rgb = rgb * 255
        rgb.astype(np.uint8)
        return rgb

    def get_random_spawn_points(self):
        _spawn_points = self.map.get_spawn_points()
        if self.random_route:
            while True:
                # start = time.time()
                spawn_ids = random.sample(range(len(_spawn_points)), 2)
                self.origin, self.destin = _spawn_points[spawn_ids[0]].location, _spawn_points[spawn_ids[-1]].location
                route_trace = self.grp.trace_route(self.origin, self.destin)
                if len(route_trace) < config['max_waypoint']:
                    continue
                route_trace = route_trace[:(config['max_waypoint'])]
                self.destin = route_trace[-1][0].transform.location
                # check initial section
                v1 = np.array([np.cos(np.radians(_spawn_points[spawn_ids[0]].rotation.yaw)),
                               np.sin(np.radians(_spawn_points[spawn_ids[0]].rotation.yaw))])
                for trace in route_trace:
                    v2 = carla_vectors_to_delta_vector(trace[0].transform.location, self.origin)
                    v2_norm = np.linalg.norm(v2)
                    if v2_norm > 5:
                        break
                if np.dot(v1, v2 / v2_norm) < 0.9:
                    continue
                if np.linalg.norm(np.cross(v1, v2)) < config['lane_invasion_threshold']:
                    if config['debug_global_path']:
                        origin_pt = np.array([self.origin.x, self.origin.y])
                        line = np.array([origin_pt + k * v1 for k in np.arange(0, 10, 0.1)])
                        plt.plot(line[:, 0], line[:, 1])
                        plt.scatter(*line[-1], c='r')
                        plt.scatter(*origin_pt, c='b')
                        plt.scatter(*origin_pt + v2, c='g')
                        plt.title("{}".format(np.linalg.norm(np.cross(v1, v2))))
                        route = np.array(
                            [[trace[0].transform.location.x, trace[0].transform.location.y] for trace in route_trace])
                        plt.plot(route[:, 0], route[:, 1])
                        plt.show()
                    # print("random spawn points criteria elaptime: {:.1f} ms".format((time.time() - start) * 1000))
                    break
        else:
            o_min = np.inf
            d_min = np.inf
            o_idx = -1
            d_idx = -1
            for idx, pt in enumerate(_spawn_points):
                o_dist = pt.location.distance(carla.Location(*self.start_location))
                d_dist = pt.location.distance(carla.Location(*self.end_location))
                if o_dist < o_min:
                    o_min = o_dist
                    o_idx = idx
                if d_dist < d_min:
                    d_min = d_dist
                    d_idx = idx
            spawn_ids = [o_idx, d_idx]
            self.origin = _spawn_points[o_idx].location
            self.destin = _spawn_points[d_idx].location
            route_trace = self.grp.trace_route(self.origin, self.destin)

        self.waypoints.clear()
        for trace in route_trace:
            self.waypoints.append(trace[0])

        # other_spawn_ids = list(range(len(_spawn_points)))
        # for i, j in enumerate(spawn_ids):
        #     del other_spawn_ids[j-i]
        # other_spawn_ids = random.sample(other_spawn_ids, 20)
        # spawn_points = [_spawn_points[spawn_ids[0]]] + [_spawn_points[idx] for idx in other_spawn_ids]

        # way_ids = random.sample(range(len(self.waypoints)), int(len(self.waypoints)*self.traffic_rate))

        if config['other_location']:
            spawn_points = [_spawn_points[spawn_ids[0]]] + [
                carla.Transform(location=carla.Location(loc[0], loc[1], 0.2),
                                rotation=carla.Rotation(yaw=loc[2])) for loc in config['other_location']]
        else:
            other_spawn_ids = set()
            for i in range(len(self.waypoints)):
                for j in range(len(_spawn_points)):
                    distance = self.waypoints[i].transform.location.distance(_spawn_points[j].location)
                    o_dist = self.origin.distance(_spawn_points[j].location)
                    d_dist = self.destin.distance(_spawn_points[j].location)
                    if distance < self.spawn_bound and o_dist > 3:
                        if not j in spawn_ids:
                            other_spawn_ids.add(j)
            other_spawn_ids = random.sample(other_spawn_ids, int(len(other_spawn_ids) * self.traffic_rate))
            #         print("number of other vehicles: {}".format(len(other_spawn_ids)))
            spawn_points = [_spawn_points[spawn_ids[0]]] + [_spawn_points[idx] for idx in other_spawn_ids]
        return spawn_points

    def set_spectator(self):
        ego_trans = self.actor_info['actors'][0].get_transform()
        ego_loc = ego_trans.location
        if config['use_spectator']:
            if self.topview:
                self.world.get_spectator().set_transform(carla.Transform(
                    carla.Location(x=ego_loc.x, y=ego_loc.y, z=ego_loc.z + 25),
                    carla.Rotation(pitch=270)))
            else:
                self.world.get_spectator().set_transform(carla.Transform(
                    carla.Location(x=ego_loc.x - 10.0 * math.cos(math.radians(ego_trans.rotation.yaw)),
                                   y=ego_loc.y - 10.0 * math.sin(math.radians(ego_trans.rotation.yaw)),
                                   z=ego_loc.z + 6.0),
                    carla.Rotation(pitch=345, yaw=ego_trans.rotation.yaw)))
        else:
            self.world.get_spectator().set_transform(carla.Transform(
                carla.Location(x=(self.roi[0] + self.roi[1]) * 0.5,
                               y=(self.roi[2] + self.roi[3]) * 0.5, z=200),
                carla.Rotation(pitch=270)))

    def destroy(self):
        #         print("Start destroy actors: number of actors : {}".format(len(self.world.get_actors())))

        for actor in self.actor_info['sensors']:
            if actor is not None:
                actor.reset()
                actor.sensor.destroy()
                actor.sensor = None

        if config['use_camera'] and self.camera_sensor is not None:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if config['use_lidar'] and self.lidar_sensor is not None:
            self.lidar_sensor.destroy()
            self.lidar_sensor = None

        # for actor in self.world.get_actors().filter('sensor.*'):
        #     if actor is not None:
        #         actor.destroy()

        #         print("Before destroy actors: ID of vehicles : {}".format([actor.id for actor in self.actor_info['actors']]))

        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor is not None:
                actor.destroy()

        if len(self.world.get_actors().filter('vehicle.*')) != 0 or len(
                self.world.get_actors().filter('sensor.*')) != 0:
            self.world.tick()

        for k in self.actor_info:
            self.actor_info[k].clear()

        #         print("After destroy actors: number of actors : {}".format(len(self.world.get_actors())))

    #         print("live num of sensor: {}".format(len(self.world.get_actors().filter('sensor.*'))))
    #         print("live num of vehicle: {}".format(len(self.world.get_actors().filter('vehicle.*'))))
    #         print("live sensors: {}".format([(actor.id, actor.type_id) for actor in self.world.get_actors().filter('sensor.*') if actor.is_alive]))
    #         print("live vehicles: {}".format([(actor.id, actor.type_id) for actor in self.world.get_actors().filter('vehicle.*') if actor.is_alive]))

    def retransform_actors(self, vehicle_spawn_points):
        # Initialize the transforms for each actors
        self.actor_info['transforms'].clear()
        self.actor_info['transforms'] = vehicle_spawn_points

        for i in range(len(self.actor_info['actors'])):
            if i > 0:
                self.actor_info['actors'][i].set_autopilot(False, self.tm_port)
            # if i == 0:
            self.actor_info['actors'][i].apply_control(emergency_stop())
            self.actor_info['actors'][i].set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor_info['actors'][i].set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor_info['actors'][i].set_transform(self.actor_info['transforms'][i])

    def reset_actor_info(self, vehicle_spawn_points):
        # distroy actors except for ego vehicle
        mycar = self.actor_info['actors'].pop(0)
        mycar_bp = self.actor_info['blueprints'].pop(0)
        for actor in self.world.get_actors().filter('vehicle.tesla.model3'):
            if actor.is_alive:
                actor.destroy()
        if len(self.world.get_actors().filter('vehicle.tesla.model3')) != 0:
            self.world.tick()
        for k in self.actor_info:
            if k != 'sensors':
                self.actor_info[k].clear()

        # generate vehicle actors without ego vehicle
        blueprint_list = []
        transform_list = []
        actor_list = []
        # generate vehicle actors
        for i in range(len(vehicle_spawn_points)):
            # Set the list of actors
            if i == 0:
                blueprint_list.append(mycar_bp)
                actor_list.append(mycar)
                transform_list.append(vehicle_spawn_points[i])
                mycar.apply_control(emergency_stop())
                mycar.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                mycar.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                mycar.set_transform(vehicle_spawn_points[i])
            else:
                globals()['car{}_blueprint'.format(i)] = random.choice(
                    self.world.get_blueprint_library().filter('vehicle.tesla.model3'))
                if globals()['car{}_blueprint'.format(i)].has_attribute('color'):
                    color = '150,150,150'
                    globals()['car{}_blueprint'.format(i)].set_attribute('color', color)
                if globals()['car{}_blueprint'.format(i)].has_attribute('driver_id'):
                    driver_id = random.choice(
                        globals()['car{}_blueprint'.format(i)].get_attribute('driver_id').recommended_values)
                    globals()['car{}_blueprint'.format(i)].set_attribute('driver_id', driver_id)
                if globals()['car{}_blueprint'.format(i)].has_attribute('is_invincible'):
                    globals()['car{}_blueprint'.format(i)].set_attribute('is_invincible', 'true')
                globals()['car{}'.format(i)] = self.world.try_spawn_actor(globals()['car{}_blueprint'.format(i)],
                                                                          vehicle_spawn_points[i])
                if globals()['car{}'.format(i)] is not None:
                    actor_list.append(globals()['car{}'.format(i)])
                    transform_list.append(vehicle_spawn_points[i])
                    blueprint_list.append(globals()['car{}_blueprint'.format(i)])

        self.actor_info['actors'] = actor_list
        self.actor_info['blueprints'] = blueprint_list
        self.actor_info['transforms'] = transform_list

        if config['use_polygon']:
            # Get actors polygon list
            self.vehicle_polygons = []
            vehicle_poly_dict = self._get_actor_polygons()
            self.vehicle_polygons.append(vehicle_poly_dict)

        if self.use_graph:
            self.neighbors = {}
            self.tp_neighbors = {}

    def check_reset_correctly(self):
        # print("check retransform correctly")
        # start = time.time()
        actor_ids = set()
        for i in range(len(self.actor_info['actors'])):
            dist = self.actor_info['actors'][i].get_location().distance(self.actor_info['transforms'][i].location)
            if dist > 1.5:
                actor_ids.add(i)

        # print(bcolors.WARNING + '{} th of {} vehicle spawn point is not matched'.format(actor_ids, len(self.actor_info['actors'])) + bcolors.ENDC)
        # self.client.apply_batch([carla.command.ApplyTransform(self.actor_info['actors'][idx],
        #                             self.actor_info['transforms'][idx]) for idx in actor_ids if self.actor_info['actors'][idx].is_alive])

        for i in actor_ids:
            dist = 100
            while dist > 1.5:
                if i != 0:
                    self.actor_info['actors'][i].set_autopilot(False, self.tm_port)
                self.actor_info['actors'][i].apply_control(emergency_stop())
                self.actor_info['actors'][i].set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self.actor_info['actors'][i].set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self.actor_info['actors'][i].set_transform(self.actor_info['transforms'][i])
                time.sleep(0.01)
                dist = self.actor_info['actors'][i].get_location().distance(self.actor_info['transforms'][i].location)

        # print("reset points criteria elaptime: {:.1f} ms".format((time.time() - start) * 1000))
        # print("complete to check retransform")

    def reset(self):
        # print("CarlaNavigationEnv: reset")
        self.async_world()

        # Initialize the transforms for each actors
        vehicle_spawn_points = self.get_random_spawn_points()

        self.reset_actor_info(vehicle_spawn_points)

        self.check_reset_correctly()

        self.update_actor_info()

        self.sync_world()

        self.set_initial_wpt()

        self.set_spectator()

        # Initialize sensor
        r_dis = len(self.waypoints) * config['sampling_resolution']
        self.prev_measurement = {'r_dis': r_dis,
                                 'lat_dist': 0.,
                                 'delta_yaw': 0.,
                                 'lspeed_lat': 0.}

        self.actor_info['sensors'][0].reset()
        self.actor_info['sensors'][1].reset()

        self.social_hazard = False
        self.pred_traj = None
        self.pred_id = None
        self.collision_id = []

        for i in range(len(self.actor_info['actors'])):
            if i > 0:
                self.actor_info['actors'][i].set_autopilot(True, self.tm_port)
                self.traffic_manager.auto_lane_change(self.actor_info['actors'][i], config['use_auto_lane_change'])
                self.traffic_manager.ignore_lights_percentage(self.actor_info['actors'][i], config['ignore_lights_percentage'])
                self.traffic_manager.ignore_signs_percentage(self.actor_info['actors'][i], config['ignore_signs_percentage'])
                self.traffic_manager.vehicle_percentage_speed_difference(self.actor_info['actors'][i], np.random.rand() * config['percentage_speed_difference'])
                self.traffic_manager.distance_to_leading_vehicle(self.actor_info['actors'][i], config['distance_to_leading_vehicle'])

        # print("complete to reset")
        return self._get_obs()

    def set_initial_wpt(self):

        way_locs = np.array([[w.transform.location.x, w.transform.location.y] for w in self.waypoints])
        self.roi = [way_locs[:, 0].min() - self.spawn_bound, way_locs[:, 0].max() + self.spawn_bound,
                    way_locs[:, 1].min() - self.spawn_bound, way_locs[:, 1].max() + self.spawn_bound]
        mycar_loc = self.actor_info['actors'][0].get_location()
        self.mycar_wpt = self.grp._dao.get_waypoint(mycar_loc)

        self.cur_wpt = self.waypoints.popleft()

        self.lat_dist = 0
        self.delta_yaw = 0
        self.t_pt = np.array([0, 0])

        self.episode_distance = 0

        self.inter_dist = 100
        self.traffic_light = 0
        self.numoflane = 1
        self.curlane = 1
        self.targetlane = 1
        #         self.behavior_planner.reset(self.waypoints)

        if config['debug_global_path']:
            bl_x, bl_y = self.roi[0], self.roi[2]
            w = self.roi[1] - self.roi[0]
            h = self.roi[3] - self.roi[2]
            rect = patches.Rectangle((bl_x, bl_y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            plt.figure(figsize=(10, 10))
            plt.plot(way_locs[:, 0], way_locs[:, 1], '.m')
            plt.plot(self.origin.x, self.origin.y, 'ro')
            plt.plot(self.destin.x, self.destin.y, 'bo')
            plt.plot(self.mycar_wpt.transform.location.x, self.mycar_wpt.transform.location.y, 'c*')
            ax = plt.gca()
            ax.add_patch(rect)
            plt.show()

    def update_way(self):
        mycar_loc = self.actor_info['actors'][0].get_location()
        self.mycar_wpt = self.grp._dao.get_waypoint(mycar_loc)
        cur_idx = self.grp._find_closest_in_list(self.mycar_wpt, list(self.waypoints)[1:min(5, len(self.waypoints))])
        for i in range(cur_idx):
            self.cur_wpt = self.waypoints.popleft()

        if config['use_roi']:
            self.update_roi()
            # draw roi
            if config['debug_roi']:
                show_roi_region(self.world, self.roi)

    def update_roi(self):
        w_loc = np.array([[w.transform.location.x, w.transform.location.y] for w in self.waypoints])
        new_roi = [w_loc[:, 0].min() - self.spawn_bound, w_loc[:, 0].max() + self.spawn_bound,
                   w_loc[:, 1].min() - self.spawn_bound, w_loc[:, 1].max() + self.spawn_bound]
        self.roi= new_roi

    def update_actor_info(self):
        dead_ids = []
        for i in range(len(self.actor_info['actors'])):
            if self.actor_info['actors'][i] is None:
                dead_ids.append(i)
                continue
            if not self.actor_info['actors'][i].is_alive:
                dead_ids.append(i)
        assert not 0 in dead_ids, "ego_vehicle is dead!!!!!!!!"

        if len(dead_ids) != 0:
            # print("{}th actor is not alive".format(i))
            for k in self.actor_info:
                if k != 'sensors':
                    for i, idx in enumerate(dead_ids):
                        if k == 'actors':
                            if self.actor_info[k][idx - i].is_alive:
                                self.actor_info[k][idx - i].destroy()
                        del self.actor_info[k][idx - i]

            # print("{}th actor was destroyed".format(i))

    def stop_all_actors(self):
        self.client.apply_batch([carla.command.SetAutopilot(actor, False, self.tm_port)
                                 for actor in self.actor_info['actors'][1:] if actor.is_alive])
        self.client.apply_batch([carla.command.ApplyTargetVelocity(actor, carla.Vector3D(0.0, 0.0, 0.0))
                                 for actor in self.actor_info['actors'] if actor.is_alive])
        self.client.apply_batch([carla.command.ApplyTargetAngularVelocity(actor, carla.Vector3D(0.0, 0.0, 0.0))
                                 for actor in self.actor_info['actors'] if actor.is_alive])
        self.client.apply_batch([carla.command.ApplyVehicleControl(actor, emergency_stop())
                                 for actor in self.actor_info['actors'] if actor.is_alive])

    def step(self, action):
        done = False
        info = {}

        action = action.tolist()
        throttle, steering_wheel, brake = action
        mycar = self.actor_info['actors'][0]
        mycar.apply_control(carla.VehicleControl(throttle=throttle,
                                                 steer=steering_wheel,
                                                 brake=brake,
                                                 hand_brake=False,
                                                 reverse=False))

        self.world.tick()

        self.update_actor_info()

        # check if cars' position are outer than criteria, then reset transform all actors
        if config['use_roi']:
            # start = time.time()
            out_actor_ids = set()
            for i in range(len(self.actor_info['actors'])):
                if self.actor_info['actors'][i].is_alive:
                    if is_outROI(self.actor_info['actors'][i].get_transform().location, self.roi):
                        out_actor_ids.add(i)

            if 0 in out_actor_ids:
                done = True
                info['failure:out_of_range'] = 1
                info['bad_transition'] = True
                print("Out_of_Range!")
                # mycar_loc = carla_vector_to_numpy_vector(self.actor_info['actors'][0].get_location())
                # print('mycar: {}, roi : {}, origin : {}'.format(mycar_loc, self.roi,
                #                         carla_vector_to_numpy_vector(self.actor_info['transforms'][0].location)))
                # reward = reward - 50+

            possible_spawn_ids = set()
            for i, trans in enumerate(self.actor_info['transforms']):
                spawn = True
                # for actor in self.actor_info['actors']:
                for actor in self.world.get_actors().filter('vehicle.*'):
                    if actor.is_alive:
                        if carla_vectors_euclidean_distance(actor.get_location(),trans.location) < 2 * self.spawn_bound:
                            spawn = False
                            break
                if mycar.get_location().distance(trans.location) < (config['tracking_threshold'] + self.spawn_bound):
                    continue

                if spawn == True:
                    possible_spawn_ids.add(i)


            out_actor_ids = random.sample(out_actor_ids, min(len(possible_spawn_ids), len(out_actor_ids)))
            possible_spawn_ids = random.sample(possible_spawn_ids, len(out_actor_ids))
            # possible_spawn_ids = list(possible_spawn_ids)[:len(out_actor_ids)]
            self.client.apply_batch([carla.command.SetAutopilot(self.actor_info['actors'][idx], False, self.tm_port)
                                     for idx in out_actor_ids if idx != 0])
            self.client.apply_batch([carla.command.ApplyTargetVelocity(self.actor_info['actors'][idx],
                                                                       carla.Vector3D(0.0, 0.0, 0.0)) for idx in
                                     out_actor_ids])
            self.client.apply_batch([carla.command.ApplyTargetAngularVelocity(self.actor_info['actors'][idx],
                                                                              carla.Vector3D(0.0, 0.0, 0.0)) for idx in
                                     out_actor_ids])
            self.client.apply_batch([carla.command.ApplyTransform(self.actor_info['actors'][i],
                                                                  self.actor_info['transforms'][j]) for i, j in
                                     zip(out_actor_ids, possible_spawn_ids)])
            self.client.apply_batch([carla.command.SetAutopilot(self.actor_info['actors'][idx], True, self.tm_port)
                                     for idx in out_actor_ids if idx != 0])
            # print("check position criteria elaptime: {}".format((time.time() - start) * 1000))

        if config['use_birdeye']:
            # Append actors polygon list
            vehicle_poly_dict = self._get_actor_polygons()
            self.vehicle_polygons.append(vehicle_poly_dict)
            while len(self.vehicle_polygons) > self.max_past_step:
                self.vehicle_polygons.pop(0)

        if config['use_polygon']:
            # Append actors polygon list
            vehicle_poly_dict = self._get_actor_polygons(draw=self.debug_polygon)
            self.vehicle_polygons.append(vehicle_poly_dict)
            while len(self.vehicle_polygons) > self.max_past_step:
                self.vehicle_polygons.pop(0)

        if config['use_spectator']:
            self.set_spectator()

        # route planner
        self.update_way()

        if config['debug_way']:
            draw_waypoints(self.world, self.waypoints,
                           life_time=1.1*config['fixed_delta_seconds'],
                           color=carla.Color(config['intensity'],config['intensity'],0))

        next_obs = self._get_obs()

        if config['use_dict_obs']:
            state_obs = next_obs['feature']
        else:
            state_obs = next_obs


        ego_trans = mycar.get_transform()
        reward, done, info = self._get_reward(done, info)

        if config['debug_action']:
            text = "throttle: {:1f}\n" \
                   "steering_wheel: {:1f}\n" \
                   "brake: {:1f}\n".format(throttle, steering_wheel, brake)
            # print(text)
            draw_string(self.world, ego_trans, text)

        if config['debug_semantic']:
            # [speed, lat_dist, delta_yaw, preview_lateral_dis, preview_delta_yaw,
            # traffic_light, is_junc, is_target_lane, dist, ttc]
            text = "speed : {:2f} \n" \
                   "lat_dist : {:2f} \n" \
                   "delta_yaw : {:2f} \n" \
                   "preview_lateral_dis : {:2f} \n" \
                   "preview_delta_yaw : {:2f} \n" \
                   "traffic_light : {:2f} \n".format(*state_obs[:6])
            draw_string(self.world, ego_trans, text)
            if done:
                print(text)

        if config['debug_tracking'] and config['use_dist_ttc']:
            dist = state_obs[6:12]
            ttc = state_obs[12:18]
            text = "other_dist : {} \n" \
                   "other_ttc : {} \n".format(dist, ttc)
            draw_string(self.world, ego_trans, text)
            if done:
                if not 'success:episode' in info:
                    print(text)

        if done:
            self.stop_all_actors()

        return next_obs, reward, done, info

    def _get_reward(self, done, info):
        mycar = self.actor_info['actors'][0]
        ego_trans = mycar.get_transform()
        ego_loc = ego_trans.location
        ego_yaw = math.radians(ego_trans.rotation.yaw)

        # Success criteria:
        # if arrive at destination
        # Calculate destination_distance
        r_success = 0
        if ego_trans.location.distance(self.destin) <= self.done_epsilon:
            done = True
            info['success:episode'] = 1
            r_success = 1
            # print("Success Episode!")

        # reward for collision
        collision = False
        colhist = self.actor_info['sensors'][0].get_collision_history()
        collision_intensity = 0
        if len(colhist) > 0:
            collision_id = list(colhist.values())[-1][0]
            collision_intensity = list(colhist.values())[-1][1]
            # r_collision = - min(collision_intensity, 10000) / 10000
            collision = True
            self.actor_info['sensors'][0].reset()
            done = True
            info['bad_transition'] = True
            info['failure:collision'] = 1
            if config['debug_failure']:
                print(bcolors.HEADER + "Collision with {} / {}".format(collision_id, collision_intensity) + bcolors.ENDC)

        # reward for traffic light invasion
        r_light_out = 0
        if self.traffic_light == 2 and config['use_traffic_light']:
            if self.map.get_waypoint(ego_trans.location).is_junction:
                light_actor = mycar.get_traffic_light()
                traffic_light_loc = light_actor.get_location()

                target_vector = carla_vectors_to_delta_vector(traffic_light_loc, ego_loc)
                light_dist = np.linalg.norm(target_vector)
                forward_vector = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
                cos = np.clip(np.dot(forward_vector, target_vector) / (light_dist + 1e-8), -1., 1.)
                d_angle = math.degrees(math.acos(cos))
                if light_dist < 20 and d_angle < 90 and cos > 0:
                    r_light_out = -1
                    light_actor.set_state(carla.TrafficLightState.Green)
                    done = True
                    info['bad_transition'] = True
                    info['failure:light_invasion'] = 1
                    if config['debug_failure']:
                        print(bcolors.HEADER + 'Light invasion! light dist, angle = ({:2f},{:2f})'.format(light_dist,
                                                                                                           d_angle) + bcolors.ENDC)

        # if lane_invasion
        # reward for out of lane
        r_out = 0
        lane_invasion = False
        if abs(self.lat_dist) > config['lane_invasion_threshold']:
            invhist = self.actor_info['sensors'][1].get_invasion_history()
            if len(invhist) > 0:
                if self.actor_info['sensors'][1].offroad > 0:
                    lane_invasion = True
                    # print(bcolors.HEADER + "off road" + bcolors.ENDC)
            if self.numoflane > 1:
                left_thres = config['lane_invasion_threshold'] + 2 * config['lane_invasion_threshold'] * (self.targetlane - 1)
                right_thres = config['lane_invasion_threshold'] + 2 * config['lane_invasion_threshold'] * (self.numoflane - self.targetlane)
                left_margin = left_thres - self.lat_dist
                right_margin = right_thres + self.lat_dist
                if left_margin < 0 or right_margin < 0:
                    lane_invasion = True
                    # print(bcolors.HEADER + "off multi-lane" + bcolors.ENDC)
                else:
                    lane_invasion = False
            else:
                lane_invasion = True
                # print(bcolors.HEADER + "off lane case 4" + bcolors.ENDC)
        # print(f"lat_dist : {self.lat_dist:<10.1f}, cur_lane : {self.curlane:<10.0f}, tgt_lane : {self.targetlane:<10.0f}, num_lane : {self.numoflane:<10.0f}, invasion : {lane_invasion}")

        if lane_invasion:
            done = True
            r_out = -1.
            self.actor_info['sensors'][1].reset()
            info['bad_transition'] = True
            info['failure:lane_invasion'] = 1
            if config['debug_failure']:
                print(bcolors.HEADER + "Lane invasion! Distance : {:1f}".format(self.lat_dist) + bcolors.ENDC)

        # mycar information
        steer = mycar.get_control().steer

        # longitudinal speed
        v = mycar.get_velocity()
        lspeed = np.array([v.x, v.y])
        cur_wpt_yaw = math.radians(self.cur_wpt.transform.rotation.yaw)
        w = np.array([math.cos(cur_wpt_yaw), math.sin(cur_wpt_yaw)])
        lspeed_lon = np.dot(lspeed, w)

        # latitudinal speed
        lspeed_lat = np.cross(w, lspeed)  # when ego is left : + , right: -

        # longitudal accerlation
        acc = mycar.get_acceleration()
        lacc = np.array([acc.x, acc.y])
        lacc = np.dot(lacc, w)

        # cost for too fast
        r_fast = 0
        speed_limit = mycar.get_speed_limit()  # constant speed limit
        max_lspeed_lon = speed_limit * 2.0 / 3.6
        if lspeed_lon > max_lspeed_lon:
            r_fast = -1
            # done = True
            # info['bad_transition'] = True
            # info['failure:speed_limit'] = 1
            # print("Speed invasion!")
            # print('limited speed : {:2f}, curremt_speed : {}'.format(speed_limit, lspeed_lon))

        r_dis = len(self.waypoints) * config['sampling_resolution']
        diff_dist = self.prev_measurement['r_dis'] - r_dis
        self.episode_distance += diff_dist
        info['episode_distance'] = self.episode_distance

        # custom_cost
        collision_weight = max(50 * min(collision_intensity, 10000) / 10000, 1)
        c_collision = collision_weight * collision + 1. * self.social_hazard
        info['constraint_cost'] = c_collision

        # custom reward
        if config['dense_reward']:
            abs_lspeed = max(abs(lspeed_lon), 0.2)
            r_success = 10 * r_success
            r_vlon = 1. * abs(lspeed_lon)
            r_vlat = - 1. * abs(lspeed_lat)
            r_lat = - 0.2 * abs(self.lat_dist)
            r_lat_diff = 10. * (abs(self.prev_measurement['lat_dist']) - abs(self.lat_dist))
            r_steer = - 0.5 * abs(steer) * abs_lspeed
            r_yaw = - 0.5 * abs(self.delta_yaw) * abs_lspeed
            reward = r_success + r_vlon + r_vlat + r_lat + r_lat_diff + r_steer + r_yaw - self.penalty * c_collision

        else:
            r_success = 100 * r_success
            r_dist = 5.0 * diff_dist
            r_out = 10 * r_out
            r_light_out = 10 * r_light_out
            r_fast = 10 * r_fast
            r_collision = - 50 * collision
            reward = r_success + r_dis + r_collision + r_out + r_light_out + r_fast

        if config['debug_reward']:
            if config['dense_reward']:
                text = f'r_vlon: {r_vlon:<10.1f}\n'
                text += f'r_vlat: {r_vlat:<10.1f}\n'
                text += f'r_lat: {r_lat:<10.1f}\n'
                text += f'r_steer: {r_steer:<10.1f}\n'
                text += f'r_yaw: {r_yaw:<10.1f}\n'
            else:
                text = f'r_success: {r_success:<10.1f}\n'
                text = f'r_dist: {r_dist:<10.1f}\n'
                text += f'r_collision: {r_collision:<10.1f}\n'
                text += f'r_out: {r_out:<10.1f}\n'
                text += f'r_light_out: {r_light_out:<10.1f}\n'
                text += f'r_fast: {r_fast:<10.1f}\n'

            draw_string(self.world, ego_trans, text)

        self.prev_measurement = {'r_dis': r_dis,
                                 'lat_dist': self.lat_dist,
                                 'delta_yaw': self.delta_yaw,
                                 'lspeed_lat': lspeed_lat}

        return reward, done, info

    def _get_preview_lane_dis(self, x, y, distance=10):
        """Calculate distance from (x, y) to waypoints."""
        index = min(int(distance / config['sampling_resolution']), len(self.waypoints) - 1)
        w = self.waypoints[index]
        waypt = np.array([w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw])

        vec = np.array([x - waypt[0], y - waypt[1]])
        w_vec = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
        dis = np.cross(w_vec, vec)
        return dis, w_vec

    def _get_lookahead_info(self, ego_trans, v):

        # local plan
        idx = int(config['max_lookahead'] / config['sampling_resolution']) + config['target_preview_index']
        local_plan = np.array([[w.transform.location.x,
                                w.transform.location.y,
                                w.transform.location.z] for w in (([self.cur_wpt] + list(self.waypoints)))[:idx]])
        local_plan = carla_global_to_local_tranform_points(ego_trans, local_plan, rot=False)[:, :2]
        local_plan = np.unique(local_plan, axis=0)

        # current point
        loc_pt = np.array([0, 0])
        ego_direction = np.array([1, 0])
        neg_indices = np.where(np.dot(local_plan, ego_direction) < 0)[0]
        pos_indices = np.where(np.dot(local_plan, ego_direction) >= 0)[0]
        if len(neg_indices) != 0 and len(pos_indices) != 0:
            top2 = [neg_indices[-1], pos_indices[0]]
        else:
            distances = np.linalg.norm(local_plan, axis=1)
            top2 = sorted(sorted(np.arange(len(local_plan)), key=lambda ind: distances[ind])[:2])
        c_pt = local_plan[top2[0]]
        n_pt = local_plan[top2[1]]

        cur_direction = n_pt - c_pt
        cur_direction = cur_direction / np.linalg.norm(cur_direction)

        cur_ego_vec = loc_pt - c_pt
        c_pt = c_pt + np.dot(cur_ego_vec, cur_direction) * cur_direction
        cur_ego_vec = loc_pt - c_pt

        # lookahead distance
        if config['dynamic_lookahead']:
            self.lat_dist = np.cross(cur_direction, cur_ego_vec)  # when ego is left : + , right: -
            A, B, C, D, E = self.vehicle_config
            lookahead = (1 / (2 * A)) * (v ** 2) + B * v + (D / math.tan(C)) + E * abs(self.lat_dist)
            lookahead = min(lookahead, config['max_lookahead'])
        else:
            lookahead = config['lookahead']

        # target point
        dist = 0
        btmp_pt = c_pt
        target_direction = cur_direction
        for w_pt in local_plan[top2[1]:]:
            ftmp_pt = w_pt
            c2t_vec = ftmp_pt - btmp_pt
            if np.dot(cur_direction, c2t_vec) > -0.9:
                dist += np.linalg.norm(c2t_vec)
            else:
                dist -= np.linalg.norm(c2t_vec)

            if dist > lookahead:
                t_pt_yaw = np.arctan2(c2t_vec[1], c2t_vec[0])
                target_direction = np.array([np.cos(t_pt_yaw), np.sin(t_pt_yaw)])
                break
            btmp_pt = ftmp_pt

        if config['interpolation']:
            dist_gap = dist - lookahead
            point_dist = max(1e-8, np.linalg.norm(ftmp_pt - btmp_pt))
            ratio = max(0, min(1, dist_gap / point_dist))
            self.t_pt = ratio * btmp_pt + (1 - ratio) * ftmp_pt
            dist = dist - dist_gap
        else:
            self.t_pt = ftmp_pt

        if config['debug_target']:
            ego_x = ego_trans.location.x
            ego_y = ego_trans.location.y
            ego_z = ego_trans.location.z
            ego_yaw = np.radians(ego_trans.rotation.yaw)

            matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], [np.sin(ego_yaw), -np.cos(ego_yaw)]])
            local_2_global = lambda x: np.matmul(matrix, x) + np.array([ego_x, ego_y])
            self.world.debug.draw_point(location=carla.Location(*local_2_global(c_pt), z=ego_z + 2.2), size=0.15,
                                        color=carla.Color(r=0, g=0, b=255),
                                        life_time=1.1*config['fixed_delta_seconds'])
            self.world.debug.draw_point(location=carla.Location(*local_2_global(self.t_pt), z=ego_z + 0.2), size=0.15,
                                        color=carla.Color(r=0, g=0, b=255),
                                        life_time=1.1*config['fixed_delta_seconds'])

        # current lateral distance and delta_yaw
        cur_direction = self.t_pt - c_pt
        cur_direction = cur_direction / np.linalg.norm(cur_direction)
        self.lat_dist = np.cross(cur_direction, cur_ego_vec)  # when ego is left : + , right: -
        self.delta_yaw = -np.arcsin(np.cross(cur_direction, ego_direction)) # # clockwise : + , counter-clockwise: -

        # target_lateral distance and target_delta_yaw
        target_ego_direction = loc_pt - self.t_pt
        target_lat_dist = np.cross(target_direction, target_ego_direction)
        target_delta_yaw = -np.arcsin(np.cross(target_direction, ego_direction))

        if config['debug_lateral_error']:
            text = "Lookahead: {:.1f}\n" \
                   "LD: {:.1f}, DY: {:.1f}\n" \
                   "TLD: {:.1f}, TDY: {:.1f}\n".format(
                dist, self.lat_dist, self.delta_yaw, target_lat_dist, target_delta_yaw)
            # print(text)
            draw_string(self.world, ego_trans, text)

        return target_lat_dist, target_delta_yaw

    def _get_obs(self):
        screen_cnt = 0
        if config['use_camera']:
            # Display camera image
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = self._rgb_to_display_surface(camera)
            if config['pygame_render']:
                self.screen.blit(camera_surface, (self.display_size * screen_cnt, 0))
                screen_cnt +=1

            if config['use_camera_obs']:
                # Prepare camara observation
                camera = pygame.surfarray.array3d(self.screen)
                camera = camera[:self.display_size, :, :]
                camera = self._display_to_rgb(camera)

        if config['use_birdeye']:
            # Birdeye rendering
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            target_ways = np.array([[w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw]
                                    for i, w in enumerate(self.waypoints) if i < 20])
            self.birdeye_render.waypoints = target_ways

            # birdeye view with roadmap and actors
            birdeye_render_types = ['roadmap', 'actors']
            if self.display_route:
                birdeye_render_types.append('waypoints')
            birdeye_surface = pygame.Surface((self.display_size, self.display_size)).convert()
            self.birdeye_render.render(birdeye_surface, birdeye_render_types)
            if config['use_birdeye_obs'] or self.display_route:
                birdeye = pygame.surfarray.array3d(birdeye_surface)
                birdeye = birdeye[:self.display_size, :, :]
                birdeye = self._display_to_rgb(birdeye)
            if config['pygame_render']:
                self.screen.blit(birdeye_surface, (self.display_size * screen_cnt, 0))
                screen_cnt += 1

        if config['use_lidar']:
            ## Lidar image generation
            # Get point cloud data
            point_cloud = np.array([[location.point.y, -location.point.x, location.point.z] for location in self.lidar_data])

            # Separate the 3D space to bins for
            # point cloud, x and y is set according to self.lidar_bin,
            # and z is set to be two bins.
            y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
            x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
            z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]

            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
            lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)

            # Add the waypoints to lidar image
            if self.display_route:
                # waypt_surface = pygame.Surface((self.display_size, self.display_size)).convert()
                # self.birdeye_render.render(waypt_surface, ['waypoints'])
                # wayptimg = pygame.surfarray.array3d(waypt_surface)
                # wayptimg = wayptimg[:self.display_size, :, :]
                # wayptimg = self._display_to_rgb(wayptimg)
                # wayptimg = (wayptimg[:, :, 0] <= 10) * (wayptimg[:, :, 1] <= 10) * (wayptimg[:, :, 2] >= 240)
                wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
            else:
                wayptimg = np.zeros_like(lidar[:,:,0])
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(np.rot90(wayptimg, 3))

            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)
            lidar = np.flip(lidar, axis=1)
            lidar = np.rot90(lidar, 1)
            lidar = lidar * 255

            # Display lidar image
            lidar_surface = self._rgb_to_display_surface(lidar)
            if config['pygame_render']:
                self.screen.blit(lidar_surface, (self.display_size * screen_cnt, 0))

        if config['pygame_render']:
            # Display on pygame
            pygame.display.flip()

        # my car info
        mycar = self.actor_info['actors'][0]
        ego_trans = mycar.get_transform()
        ego_velocity = np.linalg.norm(carla_vector_to_numpy_vector(mycar.get_velocity()))
        preview_lateral_dis, preview_delta_yaw = self._get_lookahead_info(ego_trans, ego_velocity)
        speed = ego_velocity * 3.6  # km / h
        # self.inter_dist = get_intersection_distance(self.cur_wpt, config['sampling_resolution'])
        # is_junc = 1 if self.mycar_wpt.is_junction else 0
        # self.traffic_light = get_traffic_light(mycar)
        self.numoflane, self.curlane = get_lane_info(self.mycar_wpt)
        self.targetlane = get_target_lane(self.waypoints, self.cur_wpt, config['lookahead'])
        # is_target_lane = 1 if self.curlane == self.targetlane else 0
        # speed_limit = mycar.get_speed_limit()

        if config['use_light_change']:
            self.change_light(mycar)

        feature = np.array([speed, self.lat_dist, self.delta_yaw, preview_lateral_dis, preview_delta_yaw])

        if config['use_traffic_light']:
            feature = np.array([*feature, self.traffic_light])

        # other car info
        if self.use_graph:
            self.neighbors = self.graph.get_neighbors(self.actor_info['actors'], self.neighbors)

            graph_feature, adjacency = self.graph.node_n_adacency(self.neighbors) # (C, T, V), # (L + 1, V, V)

            self.tp_neighbors = self.tp_graph.get_neighbors(self.actor_info['actors'], self.tp_neighbors)
        if self.use_tracking:
            neighbor_actors = get_neighbor_actors(self.actor_info['actors'],
                                                  threshold=config['tracking_threshold'],
                                                  numofobj=config['tracking_number'])
            tracking_info = get_tracking_info(mycar,
                                              neighbor_actors,
                                              config['tracking_number'])

        if config['debug_predict_trajectory']:
            if self.pred_traj is not None:
                vis_trajectory(self.world, ego_trans, self.pred_traj, self.pred_id, config['debug_id'],
                               1.1*config['fixed_delta_seconds'], carla.Color(0,config['intensity'],0))

        if config['use_dict_obs']:
            next_obs = {'feature': feature}
            if config['use_camera_obs']:
                next_obs.update({'camera': camera})
            if config['use_birdeye_obs']:
                next_obs.update({'birdeye': birdeye})
            if config['use_lidar_obs']:
                next_obs.update({'lidar': lidar})
            if self.use_graph:
                next_obs.update({'graph_feature': graph_feature,
                                 'adjacency': adjacency})
            if self.use_tracking:
                next_obs.update({'tracking': tracking_info})
        else:
            next_obs = feature

        return next_obs

    def change_light(self, mycar):
        self.dt = time.time() - self.prev_time
        if self.traffic_light == 2:
            self.light_elapse_time += self.dt
            # print('elapse_time :{}, dt : {}'.format(self.elapse_time, dt))
            if self.light_elapse_time > self.red_light_time:
                mycar.get_traffic_light().set_state(carla.libcarla.TrafficLightState.Green)
                self.traffic_light = get_traffic_light(mycar)
                # print("change traffic light from red into green")
                self.light_elapse_time = 0
        self.prev_time = time.time()

    def set_hazard(self, hazard, predicted_trajectory, pred_id, collision_id):
        self.social_hazard = hazard
        self.pred_traj = predicted_trajectory
        self.pred_id = pred_id
        self.collision_id = collision_id

    def get_vehicle_polygons(self):
        return list(self.vehicle_polygons[-1].values())
    def get_polygon_keys(self):
        return np.array(list(self.vehicle_polygons[-1].keys()))
    def get_neighbor_keys(self):
        return np.array(list(self.neighbors.keys()))

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.traffic_manager.set_random_device_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def sync_world(self):
        # make world synchronous
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = config['no_rendering_mode']
        settings.quality_level = config['quality_level']
        settings.fixed_delta_seconds = config['fixed_delta_seconds']
        settings.max_substep_delta_time = config['max_substep_delta_time']
        settings.max_substeps = config['max_substeps']
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)

    def async_world(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(False)

    def close(self):
        #         print("CarlaNavigationEnv: close")
        # print("close environment...")
        if self.world is not None:
            self.async_world()
            self.destroy()

        if (config['use_camera'] or config['use_birdeye'] or config['use_lidar']) and config['pygame_render']:
            # Close pygame window
            pygame.quit()

        print("Close environment: number of actors : {}".format(len(self.world.get_actors())))
        # print(os.system('lsof -i TCP:2000 | wc -l'))
