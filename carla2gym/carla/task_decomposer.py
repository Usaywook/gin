import carla
import numpy as np
import math
from carla2gym.carla.transform import *
from carla2gym.carla.util import draw_string,showVector

class TaskDecomposer():
    def __init__(self, carla_world, num_of_task=5):
        self.world = carla_world
        self.map = carla_world.get_map()
        self.num_of_task = num_of_task
        self.current_task = 0
        self.target = 'follow'
        self.traffic_lights =  [actor for actor in carla_world.get_actors() if actor.type_id =='traffic.traffic_light']

    def get_current_task(self, mycar, waypoints, is_onehot=False):

        waypoint = self.map.get_waypoint(mycar.get_location(),
                                         project_to_road=True,
                                         lane_type=carla.LaneType.Driving)

        self.current_task = 0
        if waypoint:
            self.decide_inter_type(mycar, waypoint)
            if self.is_lane_change(waypoint, waypoints):
                self.current_task = 1

        if is_onehot:
            temp = np.zeros(self.num_of_task)
            temp[self.current_task] = 1
            return temp

        return self.current_task

    def is_lane_change(self, waypoint, waypoints, thres=4):
        self.target = 'follow'
        if waypoint.lane_change:
            if len(waypoints) > 4:
                cur = waypoint
                next = waypoints[3]
                cur_lane_width = cur.lane_width
                cur_yaw = cur.transform.rotation.yaw
                next_yaw = next[2]
                next_loc = carla.Location(x=next[0], y=next[1])

                cur_vector = np.array([math.cos(math.radians(cur_yaw)),math.sin(math.radians(cur_yaw))])
                cur_next_vector = carla_vectors_to_delta_vector(next_loc, cur.transform.location)
                lane_distance = np.cross(cur_vector, cur_next_vector)
                delta_yaw = abs(cur_yaw - next_yaw)
                v_dist = 15 * abs(math.tan(math.radians(delta_yaw)))
                if cur_lane_width / 2 < abs(lane_distance) < cur_lane_width + thres:
                    if v_dist < 0.1:
                        if lane_distance > 0:
                            self.target = 'right'
                        else:
                            self.target = 'left'
                        return True
        return False

    def decide_inter_type(self, mycar, waypoint):

        if waypoint.is_junction:
            is_light = False
            for light in self.traffic_lights:
                light_dist = carla_vectors_euclidean_distance(mycar.get_location(), light.get_location())
                if light_dist < 30:
                    self.current_task = 2
                    is_light = True
                    break
            if not is_light:
                self.current_task = 3
                xy = carla_vector_to_numpy_vector(waypoint.transform.location)
                if np.linalg.norm(xy) < 30:
                    self.current_task = 4

    def show_task(self, ego_trans=None, x=3.0, y=0.0, z=10.0):
        tasks = ['lane_following', 'lane_changing', 'signed_intersection', 'unsigned_intersection', 'roundabout']
        text = tasks[self.current_task]
        if not ego_trans:
            ego_trans = carla.Transform(location=carla.Location(x,y))

        draw_string(self.world, ego_trans, text, z=z, color = carla.Color(r=0,g=255,b=0))

        # vector = np.zeros(self.num_of_task)
        # vector[self.current_task] = 1
        # showVector(self.world, vector, x=x, y=y, z=z)

    def get_target_change(self):
        return self.target