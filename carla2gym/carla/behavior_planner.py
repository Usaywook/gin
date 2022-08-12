#!/usr/bin/python
import random
import numpy as np
import pdb
import copy
from collections import deque

from carla2gym.carla.transform import *
from carla2gym.carla.util import *

def get_lat_dist(wc, wn):
    c_yaw = wc.transform.rotation.yaw
    v1 = np.array([math.cos(math.radians(c_yaw)),
                   math.sin(math.radians(c_yaw))])
    v2 = carla_vectors_to_delta_vector(wn.transform.location, wc.transform.location)
    lat_dist = np.cross(v1,v2)
    return lat_dist

def get_idxs(wpt_q):
    right_idxs = [i for i in range(len(wpt_q) - 1) if get_lat_dist(wpt_q[i], wpt_q[i + 1]) > 0.01
                                                    and wpt_q[i].is_junction]
    left_idxs = [i for i in range(len(wpt_q) - 1) if get_lat_dist(wpt_q[i], wpt_q[i + 1]) < -0.01
                                                    and wpt_q[i].is_junction]
    lc_idxs = [i for i in range(len(wpt_q)) if str(wpt_q[i].lane_change) != 'NONE']
    junc_idxs = [i for i in range(len(wpt_q)) if wpt_q[i].is_junction]

    return {'right':right_idxs, 'left':left_idxs, 'lc':lc_idxs, 'junc':junc_idxs}

def get_next_idx(c_idx, idxs):
    min = 1000
    min_idx = 1000
    for idx in idxs:
        diff = idx - c_idx
        if diff > 0 and diff < min:
            min = diff
            min_idx = idx
    return min_idx

def is_turn_allowed(c_wpt):
    n_wpts = c_wpt.next(4.5)
    if n_wpts:
        for w in n_wpts:
            lat_dist = get_lat_dist(c_wpt, w)
            if -0.1 < lat_dist < 0.1:
                continue
            return True
    return False

def get_turn_at_inter(junc_idx, r_turn_idx, l_turn_idx):
    if r_turn_idx < l_turn_idx:
        if abs(junc_idx - r_turn_idx) < 5:
            return True, 'right'
    if l_turn_idx < r_turn_idx:
        if abs(junc_idx - l_turn_idx) < 5:
            return True, 'left'
    return False, 'follow'

def is_end_lane(c_wpt, check_distance=4.5*3):
    if c_wpt.next(check_distance) is None:
        return True
    return False

def is_target_lane(numoflane, curlane, targetlane, turn_target):
    if curlane != targetlane:
        if curlane > targetlane:
            change_target = 'left'
        elif curlane < targetlane:
            change_target = 'right'
        return False, change_target

    if turn_target == 'right':
        if numoflane == curlane:
            return True, 'follow'
    elif turn_target == 'left':
        if curlane == 1:
            return True, 'follow'
    return False, turn_target

class BehaviorPlanner(object):

    def __init__(self, safety_distance=10, collision_distance=5, safty_time=3,
                 min_inter_change_dist = 0, max_inter_change_dist= 100, use_bypass = False, bypass_wait_cnt=20):

        self.reset()

        # parameter


        self.safety_distance = safety_distance
        self.collision_distance = collision_distance
        self.safty_time = safty_time

        self.min_inter_change_dist = min_inter_change_dist
        self.max_inter_change_dist = max_inter_change_dist
        self.use_bypass = use_bypass
        self.bypass_wait_cnt = bypass_wait_cnt

    def destroy(self):
        del self.carla_info
        del self.behaviors
        del self.idx_table
        del self.global_plan

    def reset(self, global_plan=None):
        self.carla_info = {'numoflane': None,
                           'curlane': None,
                           'inter_dist': None,
                           'is_junction': None,
                           'is_turn_allowed': None,
                           'is_end_lane': None,
                           'lane_change': None,
                           'traffic_light': None,
                           'ego_loc': None,
                           'ego_rot': None,
                           'is_front_vehicle': None,
                           'change_hazard': None}

        self.behaviors = {'follow': False,
                          'stop': False,
                          'straight': False,
                          'turn_right': False,
                          'turn_left': False,
                          'change_right': False,
                          'change_left': False}

        self.idx_table = {'right': [],
                          'left': [],
                          'lc': [],
                          'junc': []}

        self.cur_idx = 0
        self.hazard_cnt = 0

        if global_plan is not None:
            self.set_global_plan(global_plan)

    def set_global_plan(self, global_plan):
        self.global_plan = [w for w in global_plan]
        self.idx_table = get_idxs(self.global_plan)

    def update_cur_idx(self, ego_loc):
        d_min = np.inf
        for i, w in enumerate(self.global_plan):
            d = w.transform.location.distance(ego_loc)
            if d < d_min:
                self.cur_idx = i
                d_min = d

    def update_carla_info(self, ego_location, c_wpt, numoflane, curlane, targetlane, inter_dist, traffic_light, dist_ttc):

        self.update_cur_idx(ego_location)
        if c_wpt:
            self.carla_info['numoflane'] = numoflane
            self.carla_info['curlane'] = curlane
            self.carla_info['targetlane'] = targetlane
            self.carla_info['inter_dist'] = inter_dist
            self.carla_info['is_junction'] = c_wpt.is_junction
            self.carla_info['is_turn_allowed'] = True # is_turn_allowed(c_wpt) #TODO
            self.carla_info['is_end_lane'] = is_end_lane(c_wpt)
            self.carla_info['lane_change'] = str(c_wpt.lane_change)
            self.carla_info['traffic_light'] = traffic_light
            change_hazard, hazard = self.surround_hazard(self.carla_info['is_junction'], dist_ttc)
            self.carla_info['change_hazard'] = change_hazard
            self.carla_info['is_front_vehicle'] = hazard

    def surround_hazard(self, is_junc, dist_ttc):

        dist = dist_ttc[0:6]
        ttc = dist_ttc[6:12]

        hazard = False
        change_hazard = [False] * 4

        if self.carla_info['traffic_light'] != 0:
            # print(bcolors.WARNING + "Traffic Light Stop" + bcolors.ENDC)
            hazard = True

        front_margin = dist[1] < self.collision_distance * 4
        sur_margin = dist < self.collision_distance
        sur_margin[1] = np.logical_or(sur_margin[1], front_margin)
        margin_check = sur_margin
        if self.behaviors['stop']:
            if True in margin_check:
                hazard = True
                # if hazard: print(bcolors.WARNING + "Front Keep Stop" + bcolors.ENDC)
        else:
            if self.carla_info['curlane'] == self.carla_info['targetlane']:
                if is_junc:
                    hazard_check = np.logical_and(dist < self.safety_distance, ttc < self.safty_time)
                else:
                    hazard_check = np.logical_and(dist < self.collision_distance * 2, ttc < self.safty_time)
                f_hazard_check = np.full(6, False, dtype=bool)
                f_hazard_check[1] = np.logical_and(dist[1] < self.safety_distance, ttc[1] < self.safty_time)
                f_hazard_check = np.logical_or(hazard_check, f_hazard_check)
                hazard = True in np.logical_or(f_hazard_check, margin_check)
                # if hazard: print(bcolors.WARNING + "Lane Following HAZARD" + bcolors.ENDC)
            else:
                hazard_check = np.logical_and(dist < self.safety_distance, ttc < self.safty_time)
                margin_check = np.logical_or(dist < self.collision_distance * 2, margin_check)
                hazard = True in np.logical_or(hazard_check, margin_check)
                change_hazard = [hazard_check[0], hazard_check[2], hazard_check[3], hazard_check[5]]
                # if change_hazard[0]:
                #     print(bcolors.WARNING + "Front Left Change HAZARD" + bcolors.ENDC)
                # if change_hazard[1]:
                #     print(bcolors.WARNING + "Front Right Change HAZARD" + bcolors.ENDC)
                # if change_hazard[2]:
                #     print(bcolors.WARNING + "Back Left Change HAZARD" + bcolors.ENDC)
                # if change_hazard[3]:
                #     print(bcolors.WARNING + "Back Right Change HAZARD" + bcolors.ENDC)

        return change_hazard, hazard

    def run(self):
        self.set_behavior()
        return self.get_behavior()

    def get_behavior_key(self):
        candidate = [k for k, v in self.behaviors.items() if v]
        assert len(candidate) == 1
        return candidate[0]

    def get_behavior(self):
        behavior = [0] * 7
        key = self.get_behavior_key()
        if key == 'follow':
            behavior[0] = 1
        elif key == 'stop':
            behavior[1] = 1
        elif key == 'straight':
            behavior[2] = 1
        elif key == 'turn_right':
            behavior[3] = 1
        elif key == 'turn_left':
            behavior[4] = 1
        elif key == 'change_right':
            behavior[5] = 1
        elif key == 'change_left':
            behavior[6] = 1
        return behavior

    def set_behavior(self):

        self.behaviors = {'follow': False,
                          'stop': False,
                          'straight': False,
                          'turn_right': False,
                          'turn_left': False,
                          'change_right': False,
                          'change_left': False}

        if self.is_hazard():
            self.behaviors['stop'] = True
            return
        direction = self.get_turn_direction()
        if direction == 'right':
            self.behaviors['turn_right'] = True
            return
        elif direction == 'left':
            self.behaviors['turn_left'] = True
            return
        elif direction == 'straight':
            self.behaviors['straight'] = True
            return
        change_direction = self.get_change_direction()
        # TOOD: other vehicle collision check
        if change_direction != 'follow':
            if self.is_collision(change_direction):
                if not self.carla_info['is_front_vehicle']:
                    change_direction = 'follow'
                else:
                    self.behaviors['stop'] = True
                    return
        if change_direction == 'right':
            # print("change right!")
            self.behaviors['change_right'] = True
            return
        elif change_direction == 'left':
            # print("change left!")
            self.behaviors['change_left'] = True
            return
        self.behaviors['follow'] = True
        return

    def is_collision(self, change_target):
        left_front, right_front, left_back, right_back = self.carla_info['change_hazard']
        if change_target == 'right':
            if right_front:
                return True
            elif right_back:
                return True
        elif change_target == 'left':
            if left_front:
                return True
            elif left_back:
                return True
        return False

    def get_change_direction(self):
        curlane = self.carla_info['curlane']
        targetlane = self.carla_info['targetlane']
        if curlane != targetlane:
            if curlane > targetlane:
                change_target = 'left'
            elif curlane < targetlane:
                change_target = 'right'
        else:
            change_target = 'follow'
        return change_target

        # TODO
        # junc_idx = get_next_idx(self.cur_idx, self.idx_table['junc'])
        # r_turn_idx = get_next_idx(self.cur_idx, self.idx_table['right'])
        # l_turn_idx = get_next_idx(self.cur_idx, self.idx_table['left'])
        # # print("junc_idx: {}, r_turn_idx: {}, l_turn_idx: {}".format(junc_idx, r_turn_idx, l_turn_idx))
        # if not self.cur_idx in self.idx_table['lc']:
        #     return 'follow'
        #
        # # will turn and which direction?
        # is_turn, turn_target = get_turn_at_inter(junc_idx, r_turn_idx, l_turn_idx)
        # # print("will turn:{}, which direciton: {}, inter_dist: {}".format(is_turn, turn_target, self.carla_info['inter_dist']))
        # if is_turn and self.carla_info['inter_dist'] < self.max_inter_change_dist:
        #     is_inter = self.carla_info['inter_dist'] <= self.min_inter_change_dist
        #     # print("is_inter: {}, inter dist: {:2f}".format(is_inter, self.carla_info['inter_dist']))
        #     if not is_inter:
        #         is_target, change_target = is_target_lane(self.carla_info['numoflane'],
        #                                                   self.carla_info['curlane'],
        #                                                   self.carla_info['targetlane'],
        #                                                   turn_target)
        #         # print("is_target: {}, curlane: {}, targetlane: {}, change_target: {}".format(is_target,
        #         #                                                                              self.carla_info['curlane'],
        #         #                                                                              self.carla_info['targetlane'],
        #         #                                                                              change_target))
        #         if not is_target:
        #             return change_target
        #
        # lane_change = self.carla_info['lane_change']
        # # end lane
        # if self.carla_info['is_end_lane']:
        #     assert lane_change != 'NONE'
        #     if lane_change == 'Both':
        #         return 'left'
        #     else:
        #         return lane_change.lower()
        #
        # if self.use_bypass and self.hazard_cnt == self.bypass_wait_cnt:
        #     print("bypass!")
        #     if lane_change == 'Both':
        #         return 'left'
        #     else:
        #         return lane_change.lower()
        #
        # return 'follow'

    def get_turn_direction(self):

        if self.carla_info['is_junction']:
            if self.carla_info['is_turn_allowed']:
                if self.cur_idx in self.idx_table['right']:
                    return 'right'
                elif self.cur_idx in self.idx_table['left']:
                    return 'left'
            return 'straight'
        return 'follow'

    def is_hazard(self):
        hazard = False
        if self.carla_info['is_front_vehicle']:
            hazard = True
            if self.use_bypass and self.hazard_cnt < self.bypass_wait_cnt:
                self.hazard_cnt += 1
        else:
            if self.use_bypass:
                self.hazard_cnt = 0

        if self.carla_info['traffic_light'] in [1,2]:
            hazard = True

        return hazard

    def show_behavior(self, world, ego_trans, z = 2):
        key = self.get_behavior_key()
        text = "{}".format(key.capitalize())
        draw_string(world, ego_trans, text, z)