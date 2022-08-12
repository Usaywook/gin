import math
import copy
import numpy as np
import torch
import copy
import carla
from carla2gym.carla.transform import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import pdb

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_other_actors(carla_world, ego_id, actor_filters):

    other_actors = []
    for actor_filter in actor_filters:
        for actor in carla_world.get_actors().filter(actor_filter):
            if actor.id != ego_id:
                if actor.is_alive:
                    other_actors.append(actor)
    return other_actors

def get_neighbor_actors(carla_actors, threshold=30, numofobj=2):
    '''
    Choose top numofobj neighbor actors that close to mycar within threshold
    '''
    neighbor = {}
    for another_actor in carla_actors[1:]:
        loc1 = carla_actors[0].get_location()
        if not another_actor.is_alive:
            continue
        loc2 = another_actor.get_location()
        distance = carla_vectors_euclidean_distance(loc2, loc1)
        if distance < threshold:
            neighbor[another_actor] = distance
    neighbor = sorted(neighbor.items(), key=lambda item: item[1])
    return [k for k,v in neighbor][:numofobj]

def get_partition_idx(loc, w):
    index = -1
    if loc[1] >= 0:
        if loc[0] <= -w / 2:
            index = 0
        elif -w / 2 < loc[0] < w / 2:
            index = 1
        elif loc[0] >= w / 2:
            index = 2
    elif loc[1] < 0:
        if loc[0] <= -w / 2:
            index = 3
        elif -w / 2 < loc[0] < w / 2:
            index = 4
        elif loc[0] >= w / 2:
            index = 5

    assert index != -1, "Get Partition Error!"

    return index

def get_ttc(obj, vec, coeff):
    def get_x_points(m, o_x, o_y, a, b, c):
        denominator = 1 + m**2
        norminator = m * np.sqrt(b**2 + a**2 / (m**2 + 1e-6))
        norminator = np.array([norminator, -norminator]) -m*o_y + m**2*o_x + c
        return norminator / denominator

    tangent = vec[1] / (vec[0] + 1e-6)
    x_cols = get_x_points(tangent, *obj, *coeff)
    col_idx = np.argmin(np.array([abs(x-obj[0]) for x in x_cols]))
    x_col = x_cols[col_idx]
    y_col = tangent * (x_col - obj[0]) + obj[1]
    p_col = np.array([x_col, y_col])
    direction = p_col - obj
    e_col = obj + (np.linalg.norm(direction) - coeff[0]) * direction / np.linalg.norm(direction)
    dist = np.linalg.norm(obj - e_col)
    ttc = dist / (np.linalg.norm(vec) + 1e-6)
    return ttc

def collision_angle_check(obj, vec, coeff=[4,2,2]):
    o_x, o_y = obj
    a, b, c = coeff
    denominator = - o_x**2 + 2*c*o_x - c**2 + a**2
    norminator = b**2*o_x**2 - 2*b**2*c*o_x + b**2*c**2 + a**2*o_y**2 - a**2*b**2
    if norminator < 0:
        return False
    norminator = np.sqrt(norminator)
    inclines = (np.array([-norminator , norminator]) - o_x*o_y + c*o_y) / denominator

    angles = [np.arctan(incline) for incline in inclines]
    diff = abs(angles[0] - angles[1]) / 2

    denom = b**2 + a**2 * inclines**2
    norm = b**2 * c - a**2 * inclines * o_y + a**2 * inclines**2 * o_x
    x_t = norm / denom
    y_t = o_y + inclines * (norm / denom - o_x)

    mid = np.array([np.sum(x_t - obj[0]), np.sum(y_t - obj[1])])
    mid = mid / np.linalg.norm(mid)
    norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
    r_loc_vec = - np.array(obj)

    on_coming = np.dot(r_loc_vec, norm_vec) > 0 # is oncoming direction
    in_colliion = abs(np.dot(norm_vec, mid)) > np.cos(diff) # is in colliion range
    collision = on_coming and in_colliion

    return collision

def get_dist_ttc(carla_actors, threshold=30, width=3.5, debug=False, coeff=[6, 2.5, 2.6]):
    # initialize the container
    neighbor = {}
    for i in range(6):
        neighbor[i] = []

    # find neighbor actors and partition index
    ego_trans = carla_actors[0].get_transform()
    ego_velocity = carla_vector_to_numpy_vector(carla_actors[0].get_velocity())
    loc1 = ego_trans.location
    ego_yaw = ego_trans.rotation.yaw
    ego_local_vel = carla_global_to_local_transform_vectors(ego_yaw + 90, np.expand_dims(ego_velocity, axis=0))[0]

    for another_actor in carla_actors[1:]:
        if another_actor.is_alive:
            loc2 = another_actor.get_location()
            distance = loc2.distance(loc1)
            if distance < threshold:
                other_global_loc = np.expand_dims(carla_location_to_numpy_vector(loc2), axis=0)
                other_local_loc = carla_global_to_local_tranform_points(ego_trans, other_global_loc)[0][:2]
                idx = get_partition_idx(other_local_loc, width)
                neighbor[idx].append((another_actor, distance, other_local_loc))

    # sort neighbor index over distance
    for k, v in neighbor.items():
        if len(v) == 0:
            continue
        neighbor[k] = sorted(v, key=lambda x: x[1])[0]

    # find ttc for actors
    for k, v in neighbor.items():
        if len(v) == 0:
            continue
        if not v[0].is_alive:
            continue
        r_loc = v[2]
        r_dist = np.linalg.norm(r_loc)
        other_global_v = carla_vector_to_numpy_vector(v[0].get_velocity())
        other_local_vel = carla_global_to_local_transform_vectors(ego_yaw + 90, np.expand_dims(other_global_v, axis=0))[0]

        r_vel = other_local_vel - ego_local_vel

        collision =  collision_angle_check(r_loc, r_vel, coeff)
        if collision:
            ttc = get_ttc(r_loc, r_vel, coeff)
        else:
            ttc = threshold

        if debug:
            neighbor[k] = (r_dist, ttc, r_loc, ego_local_vel, other_local_vel, r_vel)
        else:
            neighbor[k] = (r_dist, ttc)

    if debug:
        vis_container = []
        for k, v in neighbor.items():
            if len(v) == 0:
                continue
            vis_container.append([v[2][0], v[2][1], v[3][0], v[3][1], v[4][0], v[4][1], v[5][0], v[5][1],
                                  k, v[0], v[1]])
        vis_container = np.array(vis_container)
        if len(vis_container) != 0:
            show_ttc(vis_container[:, 0:2], vis_container[:, 2:4], vis_container[:, 4:6], vis_container[:, 6:8],
                     vis_container[:, 8], vis_container[:, 10])

    result = np.ones(12) * threshold
    for k, v in neighbor.items():
        if len(v) == 0:
            continue
        if type(v[0]) is not np.float64:
            continue
        result[k] = v[0]
        result[k + 6] = v[1]
    return result

def show_ttc(r_loc, ego_vel, other_vel, r_vel, idxs, ttc):
    f = plt.gcf()
    plt.axis("equal")
    plt.ion()
    plt.cla()
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.title("(Distance, TTC)", fontsize=30)
    plt.axhline(y=0, color='y', linestyle='--', linewidth=1)
    plt.axvline(x=-1.75, color='y', linestyle='--', linewidth=1)
    plt.axvline(x=1.75, color='y', linestyle='--', linewidth=1)

    # draw tracking result
    e_loc = np.zeros_like(r_loc)
    plt.quiver(e_loc[:, 0], e_loc[:, 1], ego_vel[:, 0], ego_vel[:, 1], color='b', scale=30)
    plt.quiver(r_loc[:, 0], r_loc[:, 1], other_vel[:, 0], other_vel[:, 1], color='r', scale=30)
    plt.quiver(r_loc[:, 0], r_loc[:, 1], r_vel[:, 0], r_vel[:, 1], color='m', scale=30)
    for loc, k, t in zip(r_loc, idxs, ttc):
        d = np.linalg.norm(loc)
        plt.text(loc[0] + 0.1, loc[1] - 0.8,"({:.1f}, {:.1f})".format(d, t), fontsize=30)

    plt.pause(0.01)

def get_dist_others(carla_actors, threshold=30, width=3.5, debug=False):
    # initialize the container
    neighbor = {}
    for i in range(6):
        neighbor[i] = []

    # find neighbor actors and partition index
    ego_trans = carla_actors[0].get_transform()
    ego_velocity = carla_vector_to_numpy_vector(carla_actors[0].get_velocity())
    loc1 = ego_trans.location
    ego_yaw = ego_trans.rotation.yaw

    for another_actor in carla_actors[1:]:
        if another_actor.is_alive:
            loc2 = another_actor.get_location()
            distance = loc2.distance(loc1)
            if distance < threshold:
                other_global_loc = np.expand_dims(carla_location_to_numpy_vector(loc2), axis=0)
                other_local_loc = carla_global_to_local_tranform_points(ego_trans, other_global_loc)[0][:2]
                idx = get_partition_idx(other_local_loc, width)
                neighbor[idx].append((another_actor, distance))

    # sort neighbor index over distance
    for k, v in neighbor.items():
        if len(v) == 0:
            continue
        neighbor[k] = sorted(v, key=lambda x: x[1])[0]

    result = np.ones(6) * threshold
    for k, v in neighbor.items():
        if len(v) == 0:
            continue
        result[k] = v[1]

    return result

def get_tracking_info(mycar_actor, neighbor_actors, max_object):
    """
    9(dx,dy,dyaw,d|v|,d|a|,d|w|) * 2(number_of_cars dim)
    tracking_info(dx, dtheta, dv, da, dw, ttc)*(# of other vehicles)
    """
    ego_trans = mycar_actor.get_transform()
    ego_yaw = np.radians(ego_trans.rotation.yaw)
    cy = np.cos(ego_yaw)
    sy = np.sin(ego_yaw)

    tracking_info = np.zeros(shape=(max_object, 9))
    for i, another_actor in enumerate(neighbor_actors):
        if another_actor.is_alive:
            info = list(carla_actor_to_delta_XYYAWVAW(another_actor, mycar_actor))
            x, y = info[:2]
            loc = [x * cy + y * sy, x * sy - y * cy]

            tracking_info[i] = [*loc, *info[2:]]

    return tracking_info

def vis_ellipsoid(world, ego_trans, coeff):
    ego_loc = ego_trans.location
    ego_yaw = np.radians(ego_trans.rotation.yaw)
    matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw), ego_loc.x],
                    [np.sin(ego_yaw), -np.cos(ego_yaw), ego_loc.y],
                    [0, 0, 1]])
    local_2_global = lambda x: np.matmul(matrix, np.vstack([x.T, np.ones((1, x.shape[0]))]))[:2].T

    ellipsoid = np.array([[coeff[0] * np.cos(theta), coeff[1] * np.sin(theta)] for theta in np.arange(0, 2 * np.pi, 0.3)])
    ellipsoid += np.array([coeff[2], 0])
    ellipsoid = local_2_global(ellipsoid)
    for ind in range(len(ellipsoid)):
        begin = carla.Location(*ellipsoid[ind], z=ego_loc.z + 0.2)
        end = carla.Location(*ellipsoid[(ind + 1) % len(ellipsoid)], z=ego_loc.z + 0.2)
        world.debug.draw_arrow(begin=begin, end=end,
                                color=carla.Color(r=0,g=255,b=255),
                                life_time=0.2, thickness=0.2)

def vis_trajectory(world, ego_trans, n_pred, neighbor_actors_id, debug_id=False, life_time=0.2, color=carla.Color(r=0,g=1,b=0)):
    ego_loc = ego_trans.location
    ego_heading = np.radians(ego_trans.rotation.yaw)
    matrix = np.array([[np.cos(ego_heading), np.sin(ego_heading), ego_loc.x],
                    [np.sin(ego_heading), -np.cos(ego_heading), ego_loc.y],
                    [0, 0, 1]])
    local_2_global = lambda x: np.matmul(matrix, np.vstack([x.T, np.ones((1, x.shape[0]))]))[:2].T
    neighbor_actors_loc_z = [actor.get_location().z for actor in world.get_actors(neighbor_actors_id)]
    for i, o_pred in enumerate(n_pred): # (num_object, T, 2)
        # TODO: local to global coordinate : flip x axis, rotate yaw, translate pos
        traj = local_2_global(o_pred) # (T, 2)
        if debug_id:
            draw_string(world,
                        carla.Transform(carla.Location(*traj[0],z=neighbor_actors_loc_z[i] + 0.2)),
                        str(neighbor_actors_id[i]))
        for ind in range(len(traj) - 1):
            begin = carla.Location(*traj[ind],z=neighbor_actors_loc_z[i] + 0.2)
            end = carla.Location(*traj[ind+1],z=neighbor_actors_loc_z[i] + 0.2)
            world.debug.draw_arrow(begin=begin, end=end,
                                    color=color,
                                    life_time=life_time,
                                    thickness=0.2)

def carla_waypoint_id_to_numpy_vector(waypoint):
    # wapoint_id = waypoint.id
    road_id = waypoint.road_id
    # section_id = waypoint.section_id
    lane_id = waypoint.lane_id
    # s = waypoint.s
    # return np.array([wapoint_id, road_id, section_id, lane_id, s])
    return np.array([road_id, lane_id])

def get_lane_info(c_wpt):

    cur_road_id = c_wpt.road_id
    cur_lane_id = c_wpt.lane_id

    r_wpts_list = []
    r_wpt = c_wpt.get_right_lane()
    if r_wpt is not None:
        target_road_id = r_wpt.road_id
        while str(r_wpt.lane_type) == 'Driving' and cur_road_id == target_road_id:
            r_wpts_list.append(r_wpt)
            r_wpt = r_wpt.get_right_lane()
            if r_wpt:
                target_road_id = r_wpt.road_id
            else:
                break

    l_wpts_list = []
    l_wpt = c_wpt.get_left_lane()
    if l_wpt is not None:
        target_road_id = l_wpt.road_id
        target_lane_id = l_wpt.lane_id
        while str(l_wpt.lane_type) == 'Driving' \
                and cur_road_id == target_road_id \
                and abs(cur_lane_id) != abs(target_lane_id):
            l_wpts_list.append(l_wpt)
            l_wpt = l_wpt.get_left_lane()
            if l_wpt:
                target_road_id = l_wpt.road_id
                target_lane_id = c_wpt.lane_id
            else:
                break
    numoflane = 1 + len(r_wpts_list) + len(l_wpts_list)
    curlane =  len(l_wpts_list) + 1
    return numoflane, curlane

def get_target_lane(waypoints, c_wpt, lookahead):
    target_idx = 0
    dist = 0
    for i,wpt in enumerate(waypoints):
        dist = wpt.transform.location.distance(c_wpt.transform.location)
        if dist > lookahead:
            target_idx = i
            break
    _, targetlane = get_lane_info(waypoints[target_idx])
    return targetlane

def find_intersection(waypoint, waypoint_separation, depth=0):
    if depth == 50:
        return waypoint.transform.location, depth
    next_waypoints = waypoint.next(waypoint_separation)
    if len(next_waypoints) > 1:
        key = 0
        for i,w in enumerate(next_waypoints):
            if w.is_junction:
                key = i
                break
        return next_waypoints[key].transform.location, depth
    elif len(next_waypoints) == 1:
        next_waypoint = next_waypoints[0]
        if next_waypoint.is_junction:
            return next_waypoint.transform.location, depth
        return find_intersection(next_waypoint, waypoint_separation, depth + 1)
    else:
        return waypoint.transform.location

def get_intersection_distance(waypoint, waypoint_separation=4.5):
    if waypoint.is_junction:
        return 0
    else:
        intersection_location, depth = find_intersection(waypoint, waypoint_separation)
        intersection_distance = (depth-1) * waypoint_separation
    return intersection_distance


def get_traffic_light(mycar_actor):
    """
    padding = 0
    carla.TrafficLightState.Red = 2
    carla.TrafficLightState.Yellow = 1
    carla.TrafficLightState.Green = 0
    """
    if mycar_actor.is_at_traffic_light():
        traffic_light = mycar_actor.get_traffic_light()

        if traffic_light.get_state() == carla.TrafficLightState.Red:
            return 2
        elif traffic_light.get_state() == carla.TrafficLightState.Yellow:
            return 1
        elif traffic_light.get_state() == carla.TrafficLightState.Green:
            return 0
        else:
            return 0
    else:
        return 0


def get_semantic_info(mycar_actor, carla_waypoint):
    """
    :returns :
    localization(x,y,yaw) + road_id, lane_id , intersection_distance, traffic_light, speed_limit # for compund task, 8 dim
    localization(x,y,yaw) + velocity(Vx,Vy), lane_angle, lane_distance # for lane following, 7 dim
    """
    mycar_transform = mycar_actor.get_transform()
    xy = carla_vector_to_numpy_vector(mycar_transform.location)
    yaw = carla_rotation_to_numpy_array(mycar_transform.rotation)

    waypoint_ids = carla_waypoint_id_to_numpy_vector(carla_waypoint) # for lane change
    intersection_distance = get_intersection_distance(carla_waypoint) # for lane change
    traffic_light = get_traffic_light(mycar_actor) # for compound task
    speed_limit = mycar_actor.get_speed_limit() # for compound task
    return np.array([*xy, yaw, *waypoint_ids, intersection_distance, traffic_light, speed_limit])

    # return np.array([*xy, yaw])

def get_lane_angle_dist(mycar_actor, cur_waypoint):

    mycar_transform = mycar_actor.get_transform()

    vxvy = carla_vector_to_numpy_vector(mycar_actor.get_velocity())
    yaw = carla_rotation_to_numpy_array(mycar_transform.rotation)
    angle = yaw - carla_rotation_to_numpy_array(cur_waypoint.transform.rotation)

    distance = carla_vectors_euclidean_distance(mycar_transform.location, cur_waypoint.transform.location)
    lane_dist = distance*math.sin(math.radians(angle))

    return (*vxvy, angle, lane_dist)

def is_outROI(carla_location, roi):
    x1, x2, y1, y2 = roi
    loc = carla_vector_to_numpy_vector(carla_location)
    if (x1 <= loc[0] <= x2) and (y1 <= loc[1] <= y2):
        return False
    else:
        return True

def is_inROI(carla_loc, origin_loc, destin_loc, bound=5):
    x_min = min(origin_loc.x, destin_loc.x) - bound
    x_max = max(origin_loc.x, destin_loc.x) + bound
    y_min = min(origin_loc.y, destin_loc.y) - bound
    y_max = max(origin_loc.y, destin_loc.y) + bound
    if (x_min < carla_loc.x < x_max) and (y_min < carla_loc.y < y_max):
        if carla_loc == origin_loc or carla_loc == destin_loc:
            return False
        else:
            return True
    else:
        return False

def is_out_roibox(loc, _roi):
    x1, x2, y1, y2 = _roi
    if (x1 <= loc[0] <= x2) and (y1 <= loc[1] <= y2):
        return False
    else:
        return True

def in_elipsoid(pos, coeff):
    return ((pos[:,0] - coeff[2])** 2 / coeff[0]**2) + (pos[:,1] ** 2 / coeff[1] ** 2) <=  1

def lane_change(current_waypoint, waypoint_distance = 9):
    # Check current lane change allowed
    if str(current_waypoint.lane_type) == 'Driving' and str(current_waypoint.lane_change) == 'Left':
        left_lane_waypoint = current_waypoint.get_left_lane()
        next_waypoints = list(left_lane_waypoint.next(waypoint_distance))
    elif str(current_waypoint.lane_type) == 'Driving' and str(current_waypoint.lane_change) == 'Right':
        right_lane_waypoint = current_waypoint.get_right_lane()
        next_waypoints = list(right_lane_waypoint.next(waypoint_distance))
    else:
        next_waypoints = list(current_waypoint.next(waypoint_distance))

    if len(next_waypoints) == 0:
        next_waypoint = current_waypoint
    elif len(next_waypoints) == 1:
        next_waypoint = next_waypoints[0]
    else:
        next_waypoint = current_waypoint
    return next_waypoint

def left_lane_change(current_waypoint, waypoint_distance = 10):
    # Check current lane change allowed
    if str(current_waypoint.lane_type) == 'Driving' and (str(current_waypoint.lane_change) == 'Left' or str(current_waypoint.lane_change) == 'Both'):
        left_lane_waypoint = current_waypoint.get_left_lane()
        if left_lane_waypoint:
            next_waypoints = list(left_lane_waypoint.next(waypoint_distance))
            if len(next_waypoints) != 0:
                next_waypoint = next_waypoints[0]
                if str(next_waypoint.lane_type) == 'Driving':
                    return next_waypoint
    return None

def right_lane_change(current_waypoint, waypoint_distance = 10):
    # Check current lane change allowed
    if str(current_waypoint.lane_type) == 'Driving' and (str(current_waypoint.lane_change) == 'Right' or str(current_waypoint.lane_change) == 'Both'):
        right_lane_waypoint = current_waypoint.get_right_lane()
        if right_lane_waypoint:
            next_waypoints = list(right_lane_waypoint.next(waypoint_distance))
            if len(next_waypoints) != 0:
                next_waypoint = next_waypoints[0]
                if str(next_waypoint.lane_type) == 'Driving':
                    return next_waypoint
    return None

def lane_follow(current_waypoint, waypoint_distance=9):
    next_waypoints = list(current_waypoint.next(waypoint_distance))
    if len(next_waypoints) == 0:
        next_waypoint = current_waypoint
    elif len(next_waypoints) == 1:
        next_waypoint = next_waypoints[0]
    else:
        next_waypoint = next_waypoints[1]
    return next_waypoint

def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.
    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0

def is_vehicle_hazard(mycar, carla_map, vehicle_list, proximity_threshold = 10.0):
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

    ego_vehicle_location = mycar.get_location()
    ego_vehicle_waypoint = carla_map.get_waypoint(ego_vehicle_location)

    for target_vehicle in vehicle_list:
        # do not account for the ego vehicle
        if target_vehicle.id == mycar.id:
            continue

        if not target_vehicle.is_alive:
            continue

        # if the object is not in our lane it's not an obstacle
        target_vehicle_waypoint = carla_map.get_waypoint(target_vehicle.get_location())
        if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            continue

        loc = target_vehicle.get_location()
        if is_within_distance_ahead(loc, ego_vehicle_location,
                                    mycar.get_transform().rotation.yaw,
                                    proximity_threshold):
            return (True, target_vehicle)

    return (False, None)

def emergency_stop():
    """
    Send an emergency stop command to the vehicle
    :return:
    """
    control = carla.VehicleControl()
    control.steer = 0.0
    control.throttle = 0.0
    control.brake = 1.0
    control.hand_brake = True

    return control

def showRoI(carla_world, mycar_transform, box):
    global_boxs = local_to_carla_global_tranform_points(mycar_transform,box)
    # local_boxs = carla_global_to_local_tranform_points(mycar_transform,global_boxs)

    for box in global_boxs:
        item = carla.Location(box[0], box[1], 0.1)
        carla_world.debug.draw_string(item, 'O', draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=0.02)
def vis_subtask(subtask, step):
    plt.ion()
    plt.cla()
    plt.xlabel('step')
    plt.ylabel('subtask')
    plt.plot(step, subtask, 'r')
    plt.pause(1e-6)

def showVector(carla_world, vector, ego_trans=None, x=3.0, y=0.0, z=10.0):
    if type(vector) == torch.Tensor:
        vector = copy.deepcopy(vector)
        temp = softmax(vector.cpu())
        vector = np.zeros_like(temp)
        vector[np.argmax(temp)] = 1

    offsets = []
    for i in range(len(vector)):
        offset = [x + i*3.0, y, z]
        offsets.append(offset)

    for i in range(len(vector)):
        if ego_trans:
            location = ego_trans.location + carla.Location(*offsets[i])
        else:
            location = carla.Location(*offsets[i])
        carla_world.debug.draw_point(location=location, size=0.1,
                                     color=carla.Color(r=0,g=0,b=int(vector[i]*255)),
                                     life_time=0.00005)

def softmax(t):
    exp_t = np.exp(t.flatten())
    exp_t_sum = exp_t.sum()
    y = exp_t/exp_t_sum
    return y.numpy()

# def showWay(waypoint, carla_world):
#     # BFS to find trajectory
#     G = nx.Graph()
#     visit = list()
#     queue = list()
#     queue.append(waypoint)
#     visit.append(waypoint)
#
#     while queue:
#         u = queue.pop()
#         next_waypoints = list(u.next(3.0))
#         if not next_waypoints:
#             break
#         else:
#             for v in next_waypoints:
#                 if v not in visit:
#                     G.add_edge(u, v)
#                     visit.append(v)
#                     queue.append(v)
#         if len(visit) > 20:
#             break
#     for u,v in list(G.edges()):
#         carla_world.debug.draw_arrow(begin=u.transform.location, end=v.transform.location,
#                                         color=carla.Color(r=0,g=0,b=255), life_time=0.001, thickness=0.5)

def get_next_waypoionts(current_waypoint, distance=3.0):

    next = []

    # Check current lane change allowed and add waypoint
    if str(current_waypoint.lane_type) == 'Driving':
        if str(current_waypoint.lane_change) == 'Right':
            right_lane_waypoint = current_waypoint.get_right_lane()
            if right_lane_waypoint:
                if str(right_lane_waypoint.lane_type) == 'Driving':
                    next += list(right_lane_waypoint.next(distance))
        elif str(current_waypoint.lane_change) == 'Left':
            left_lane_waypoint = current_waypoint.get_left_lane()
            if left_lane_waypoint:
                if str(left_lane_waypoint.lane_type) == 'Driving':
                    next += list(left_lane_waypoint.next(distance))
        elif str(current_waypoint.lane_change) == 'Both':
            right_lane_waypoint = current_waypoint.get_right_lane()
            if right_lane_waypoint:
                if str(right_lane_waypoint.lane_type) == 'Driving':
                    next += list(right_lane_waypoint.next(distance))
            left_lane_waypoint = current_waypoint.get_left_lane()
            if left_lane_waypoint:
                if str(left_lane_waypoint.lane_type) == 'Driving':
                    next += list(left_lane_waypoint.next(distance))

    return next

def is_visited(waypoint, visit, threshold=1.0):
    for visted_point in visit.keys():
        dist = carla_location_euclidean_distance(waypoint.transform.location, visted_point.transform.location)
        if dist < threshold:
            return True
    return False

def showWay(waypoint, carla_world, max_depth=20, distance=3.0, lane_change_flag = True, lc_depth=-1):
    # DFS to show graph
    # G = nx.Graph()
    edges = []
    visit = {}
    stack = [(waypoint,0)]

    while stack:
        u, depth = stack.pop()
        if depth == max_depth:
            continue
        visit[u] = True

        next_waypoints = list(u.next(distance))
        if lane_change_flag and depth > lc_depth:
            next_waypoints += get_next_waypoionts(u, distance)
            lane_change_flag = False
        # next_waypoints += get_next_waypoionts(u, distance)

        if not next_waypoints:
            continue
        else:
            for v in next_waypoints:
                if not is_visited(v,visit):
                    if depth < max_depth - 1:
                        # G.add_edge(u, v)
                        edges.append((u,v))
                    stack.append((v, depth+1))

    # for u,v in list(G.edges()):
    for u, v in list(edges):
        carla_world.debug.draw_arrow(begin=u.transform.location, end=v.transform.location,
                                        color=carla.Color(r=0,g=0,b=255), life_time=0.001, thickness=0.3)

def get_trajectory(waypoint, target_vec, max_depth=20, distance=3.0, lane_change_flag = True, lc_depth=-1):
    # DFS to find trajectory
    visit = {}
    trajectory = []
    path = [waypoint]
    stack = [(waypoint,0, path)]

    while stack:
        u, depth, path = stack.pop()
        if depth == max_depth:
            continue
        visit[u] = True

        if depth == max_depth -1:
            trajectory.append(path)

        next_waypoints = list(u.next(distance))

        if lane_change_flag and depth > lc_depth:
            next_waypoints += get_next_waypoionts(u, distance)
            lane_change_flag = False
        # next_waypoints += get_next_waypoionts(u, distance)

        if not next_waypoints:
            continue
        else:
            for v in next_waypoints:
                # if v not in visit:
                if not is_visited(v, visit):
                    _path = [p for p in path] # deepcopy
                    if depth < max_depth - 1:
                        _path.append(v)
                    stack.append((v, depth+1, _path))

    infeasible_trajectory_idxs = []
    for i, path in enumerate(trajectory):
        vec = carla_vectors_to_delta_vector(path[3].transform.location, path[0].transform.location)
        vec = vec / np.linalg.norm(vec)
        if np.dot(target_vec, vec) <= np.cos(np.radians(40)):
            infeasible_trajectory_idxs.append(i)
    for i, idx in enumerate(infeasible_trajectory_idxs):
        del trajectory[idx - i]

    return trajectory

class Node():
    """A node class for A* Path finding"""

    def __init__(self, waypoint, parent=None):

        self.waypoint = waypoint
        self.parent = parent
        self.position = waypoint.transform.location

        self.g = 0
        self.h = 0
        self.f = 0
        self.thresold = 3.0
    # 연산자 오버로딩
    def __eq__(self, other):
        dist = carla_location_euclidean_distance(self.position, other.position)
        if dist < self.thresold:
            return True
        return False

def get_target_path(start, end, sampling_radius=5.0, debug=False):
    '''Use A* Path finding'''

    # Create start and end node
    start_node = Node(start, None)
    end_node = Node(end, None)


    # Initialize both open and closed list
    open_list = []  # 탐색중인 Node가 담긴 container : heap자료구조이면 좋다. 근데 귀찮으니 그냥 list로..
    closed_list = []  # path설정이 완료된 Node

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        # openlist에 들어있는 Node 들 중 가장작은 f값을 갖는 Node를 꺼내온다.
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        if debug:
            print("[{:.3f},{:.3f}], cost: {:.3f}".format(current_node.waypoint.transform.location.x,
                                   current_node.waypoint.transform.location.y,
                                   current_node.f))

        # Found the goal
        # Node 백트랙캉하면서 위치좌표를 path에 넣고 마지막에 path에 들어간 순서를 뒤집어서 순서대로 해줌
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.waypoint)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        # children은 인접하고 길이될 수 있는 노드들의 list이다.
        children = []
        next_waypoints = list(current_node.waypoint.next(sampling_radius))
        next_waypoints += get_next_waypoionts(current_waypoint=current_node.waypoint, distance=sampling_radius)
        for new_waypoint in next_waypoints:  # Adjacent squares
            # Create new node
            new_node = Node(new_waypoint, current_node)
            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            # 경로설정이 완료된 노드는 통과
            finished = False
            for closed_child in closed_list:
                if child == closed_child:
                    finished = True
            if finished:
                continue

            # compute the f, g, and h values
            child.g = current_node.g + 5 # carla_location_euclidean_distance(current_node.position, child.position)
            child.h = carla_location_euclidean_distance(child.position, end_node.position)  # remained distance
            # child.h = carla_location_manhattan_distance(child.position, end_node.position)  # remained distance
            child.f = child.g + child.h

            # check if Child is already in the open list
            is_exist = False
            index = 0
            for i, open_node in enumerate(open_list):
                if child == open_node:
                    is_exist = True
                    index = i
            if is_exist:
                if child.g > open_list[index].g:
                    continue
            # Add the child to the open list
            # update open_list
            if is_exist:
                open_list.pop(index)
            open_list.append(child)


# def get_global_path(waypoint, destination, max_depth=20, distance=3.0, lane_change_flag=True, threshold=2.0):
#     paths = get_trajectory(waypoint, max_depth, distance, lane_change_flag)
#
#     path_idx = 0
#     way_idx = 0
#     is_find = False
#
#     for i, path in enumerate(paths):
#         for j, w in enumerate(path):
#             destination_distance = carla_location_euclidean_distance(w.transform.location, destination)
#             if destination_distance < threshold:
#                 path_idx = i
#                 way_idx = j
#                 is_find = True
#                 break
#         if is_find:
#             break
#
#     if is_find:
#         return paths[path_idx][:way_idx+2]
#     else:
#         return is_find

def get_global_path(curr_waypoint, end_waypoint, sampling_radius=5.0, debug=False):

    path = get_target_path(curr_waypoint, end_waypoint, sampling_radius, debug)

    add_way = end_waypoint

    for _ in range(4):
        add_ways = list(add_way.next(sampling_radius))
        if len(add_ways) == 1:
            add_way = add_ways[0]
            path += add_ways

    return path

def get_pos_index(angle):
    index = None
    if (-np.pi / 6 <= angle <= 0) or (0 <= angle < np.pi / 6):
        index = 0
    elif np.pi / 6 <= angle < 5 * np.pi / 6:
        index = 1
    elif (5 * np.pi / 6 < angle <= np.pi) or (-np.pi <= angle < -5 * np.pi / 6):
        index = 2
    elif (-5 * np.pi / 6 <= angle < -np.pi / 6):
        index = 3
    return index


def get_loc_index(loc, w, target):
    index = -1

    if (loc[0] >= 0) and (-w / 2 <= loc[1] < w / 2):
        index = 0
    elif (loc[0] < 0) and (-w / 2 <= loc[1] < w / 2):
        index = 3

    if target == 0:
        if (loc[0] >= 0) and (w / 2 <= loc[1] < 3 * w / 2):
            index = 1
        elif (loc[0] < 0) and (w / 2 <= loc[1] < 3 * w / 2):
            index = 2
    if target == 1:
        if (loc[0] >= 0) and (-3 * w / 2 <= loc[1] < -w / 2):
            index = 1
        elif (loc[0] < 0) and (-3 * w / 2 <= loc[1] < -w / 2):
            index = 2

    return index

def get_waypoint_distance(w1, w2):
    y1 = w1.transform.rotation.yaw
    y2 = w2.transform.rotation.yaw
    v1 = np.array([math.cos(math.radians(y1)), math.sin(math.radians(y1))])
    v2 = carla_vectors_to_delta_vector(w2.transform.location, w1.transform.location)
    way_dist = np.cross(v1, v2)
    return way_dist

def is_same_direction(w1,w2):
    y1 = w1.transform.rotation.yaw
    y2 = w2.transform.rotation.yaw
    v1 = np.array([math.cos(math.radians(y1)), math.sin(math.radians(y1))])
    v2 = np.array([math.cos(math.radians(y2)), math.sin(math.radians(y2))])
    cos_way = np.dot(v1, v2)
    return cos_way > 0

def get_surround_hazard(mycar, carla_map, vehicle_list, cur_w, waypoints, d_th = 10):
    result = np.zeros(2)
    ego_vehicle_location = mycar.get_location()
    ego_way = carla_map.get_waypoint(ego_vehicle_location)
    ego_way_yaw = ego_way.transform.rotation.yaw
    ego_way_vector = np.array([math.cos(math.radians(ego_way_yaw)), math.sin(math.radians(ego_way_yaw))])
    ego_yaw = mycar.get_transform().rotation.yaw
    ego_way_vector = carla_global_to_local_transform_vectors(ego_yaw, np.array([ego_way_vector]))[0]
    lane_width = ego_way.lane_width

    info = np.zeros(4)
    for target_vehicle in vehicle_list:
        target_way = carla_map.get_waypoint(target_vehicle.get_location())
        target_way_yaw = target_way.transform.rotation.yaw
        target_way_vector = np.array([math.cos(math.radians(target_way_yaw)), math.sin(math.radians(target_way_yaw))])
        ego_target_way_vector = carla_vectors_to_delta_vector(target_way.transform.location, ego_way.transform.location)
        target_way_vector, ego_target_way_vector = \
            carla_global_to_local_transform_vectors(ego_yaw,np.array([target_way_vector, ego_target_way_vector]))
        cos_way = np.dot(ego_way_vector,target_way_vector)
        way_dist = np.cross(ego_way_vector,ego_target_way_vector)

        if ((lane_width / 2) < abs(way_dist) < (3*lane_width / 2)) and cos_way > 0:
            loc = target_vehicle.get_location()
            target_vector = carla_vectors_to_delta_vector(loc, ego_vehicle_location)
            forward_vector = np.array([math.cos(math.radians(ego_yaw)), math.sin(math.radians(ego_yaw))])
            forward_vector,target_vector = carla_global_to_local_transform_vectors(ego_yaw, np.array([forward_vector,target_vector]))
            d = np.linalg.norm(target_vector)
            if d < d_th:
                cos = np.dot(forward_vector, target_vector)/d
                # left
                if way_dist > 0:
                    # taget forward
                    if cos >= 0:
                        info[0] = 1
                    # target back
                    else:
                        info[1] = 1
                # right
                else:
                    # taget forward
                    if cos >= 0:
                        info[2] = 1
                    # target back
                    else:
                        info[3] = 1

    if cur_w.lane_change and len(waypoints) > 4: #and v_dist < 0.1:
        # check lane_changing situation
        cur_wyaw = cur_w.transform.rotation.yaw
        # width = cur_w.lane_width
        cur_wv = np.array([math.cos(math.radians(cur_wyaw)),
                           math.sin(math.radians(cur_wyaw))])
        next_w = waypoints[3]
        # next_wyaw = next_w[2]
        # delta_yaw = abs(cur_wyaw - next_wyaw)
        next_loc = carla.Location(x=next_w[0], y=next_w[1])
        cur_next_wv = carla_vectors_to_delta_vector(next_loc, cur_w.transform.location)
        # v_dist = 15 * abs(math.tan(math.radians(delta_yaw)))

        cur_wv, cur_next_wv = \
            carla_global_to_local_transform_vectors(ego_yaw, np.array([cur_wv, cur_next_wv]))
        lane_dist = np.cross(cur_wv, cur_next_wv)
        if lane_dist > 0.1: #width / 2:
            result = info[:2]
        elif lane_dist < -0.1: #-width / 2:
            result = info[2:4]

    return result

def get_surround_dist(mycar, carla_map, vehicle_list):
    ego_vehicle_location = mycar.get_location()
    ego_way = carla_map.get_waypoint(ego_vehicle_location)
    ego_way_yaw = ego_way.transform.rotation.yaw
    ego_way_vector = np.array([math.cos(math.radians(ego_way_yaw)), math.sin(math.radians(ego_way_yaw))])
    ego_yaw = mycar.get_transform().rotation.yaw
    ego_way_vector = carla_global_to_local_transform_vectors(ego_yaw, np.array([ego_way_vector]))[0]
    lane_width = ego_way.lane_width

    info = np.ones(5)*50
    for target_vehicle in vehicle_list:
        if target_vehicle.is_alive:
            target_way = carla_map.get_waypoint(target_vehicle.get_location())
            target_way_yaw = target_way.transform.rotation.yaw
            target_way_vector = np.array([math.cos(math.radians(target_way_yaw)), math.sin(math.radians(target_way_yaw))])
            ego_target_way_vector = carla_vectors_to_delta_vector(target_way.transform.location, ego_way.transform.location)
            target_way_vector, ego_target_way_vector = \
                carla_global_to_local_transform_vectors(ego_yaw,np.array([target_way_vector, ego_target_way_vector]))
            cos_way = np.dot(ego_way_vector,target_way_vector)
            way_dist = np.cross(ego_way_vector,ego_target_way_vector)

            loc = target_vehicle.get_location()
            target_vector = carla_vectors_to_delta_vector(loc, ego_vehicle_location)
            forward_vector = np.array([math.cos(math.radians(ego_yaw)), math.sin(math.radians(ego_yaw))])
            forward_vector,target_vector = carla_global_to_local_transform_vectors(ego_yaw,
                                                                    np.array([forward_vector,target_vector]))
            d = np.linalg.norm(target_vector)
            cos = np.clip(np.dot(forward_vector, target_vector)/(d + 1e-8),-1.,1.)
            d_angle = math.degrees(math.acos(cos))

            if d_angle < 90 and abs(way_dist) < lane_width :
                info[0] = d

            if ((lane_width / 2) < abs(way_dist) < (3*lane_width / 2)) and cos_way > 0:
                # left
                if way_dist > 0:
                    # taget front
                    if cos >= 0:
                        info[1] = d
                    # target back
                    else:
                        info[2] = d
                # right
                else:
                    # taget front
                    if cos >= 0:
                        info[3] = d
                    # target back
                    else:
                        info[4] = d

    return info

def get_other_info(info, ego_v, cur_w, waypoints, th=0.1, t_th=15, d_th=10):
    result = np.zeros(4)
    delta_dist = np.linalg.norm(info[:,0],axis=1)

    cur_wyaw = cur_w.transform.rotation.yaw
    width = cur_w.lane_width
    cur_wv = np.array([math.cos(math.radians(cur_wyaw)),
                       math.sin(math.radians(cur_wyaw))])
    next_w = waypoints[3]
    next_wyaw = next_w[2]
    delta_yaw = abs(cur_wyaw - next_wyaw)
    next_loc = carla.Location(x=next_w[0], y=next_w[1])
    cur_next_wv = carla_vectors_to_delta_vector(next_loc, cur_w.transform.location)
    v_dist = 15 * abs(math.tan(math.radians(delta_yaw)))

    # ttc
    numofvehicle = len(info)
    delta_v = info[:, 1] - ego_v
    delta_speed = np.linalg.norm(delta_v, axis=1)
    d = info[:, 0] / delta_dist.reshape(numofvehicle, 1)
    v = delta_v / (delta_speed.reshape(numofvehicle, 1) + 1e-8)
    r_speed = delta_speed * np.sum(d * v, axis=1)
    ttc = -delta_dist / (r_speed + 1e-8)

    target = -1
    if cur_w.lane_change and len(waypoints) > 4 and v_dist < th:
        lane_dist = np.cross(cur_wv, cur_next_wv)
        if lane_dist > width / 2:
            target = 1
        elif lane_dist < -width /2:
            target = 0
        else:
            target = -1

    pos_idx = [get_loc_index(loc,width,target) for loc in info[:,0]]
    for i, d, t in zip(pos_idx, delta_dist, ttc):
        if d < d_th and i >= 0:
            result[i] = 1
            if i > 0 and not(abs(t) < t_th):
                result[i] = 0
    return result

def draw_waypoints(world, waypoints, z=0.5, life_time=0.2, color=carla.Color(1,0,0)):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the list of x,y,yaw
    :param z: height in meters
    :return:
    """
    for i, w in enumerate(waypoints):
        if type(w) == carla.Waypoint:
            begin = w.transform.location
            end = waypoints[min(i+1, len(waypoints)-1)].transform.location
        elif type(w) == list:
            x,y,yaw = w
            begin = carla.Location(x=x,y=y,z=z)
            angle = math.radians(yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.2, life_time=life_time, color=color, thickness=0.3)

def draw_string(world, ego_trans, text, z=0.5, color = carla.Color(r=0,g=255,b=0)):

    mycar_location = ego_trans.location
    location = mycar_location + carla.Location(x=0,y=0, z=z)
    world.debug.draw_string(location=location, text=text, color=color, life_time=0.0001)

def draw_trajectory(world, trajectory, text_depth=3):
    for i, path in enumerate(trajectory):
        for j in range(len(path) - 1):
            world.debug.draw_arrow(begin=path[j].transform.location, end=path[j + 1].transform.location,
                                            color=carla.Color(r=0, g=0, b=255), life_time=0.2, thickness=0.3)
    for i, path in enumerate(trajectory):
        location = path[text_depth].transform.location + carla.Location(x=0, y=0, z=1.0)
        world.debug.draw_string(location=location, text=str(i), color=carla.Color(r=0, g=255, b=0),
                                         life_time=0.0001)

def showOther(world, other_loc, other_vec, color=carla.Color(0,255,0), life_time=0.01):

    for loc, vec in zip(other_loc,other_vec):
        begin = carla.Location(*loc)
        end = begin + carla.Location(x=vec[0], y=vec[1])
        world.debug.draw_arrow(begin, end, arrow_size=0.2, life_time=life_time, color=color, thickness=0.2)

def show_roi_region(world, roi):
    region = [[roi[0], roi[2]], [roi[1], roi[2]],
              [roi[1], roi[3]], [roi[0], roi[3]], [roi[0], roi[2]]]
    for i in range(len(region) - 1):
        begin = carla.Location(x=region[i][0], y=region[i][1], z=3)
        end = carla.Location(x=region[i + 1][0], y=region[i + 1][1], z=3)
        world.debug.draw_line(begin, end, 0.4, carla.Color(255, 0, 0), 0.2)