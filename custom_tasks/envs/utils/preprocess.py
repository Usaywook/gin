from threading import local
import numpy as np

from collections import deque
from scipy.spatial.distance import cdist

from custom_tasks.envs.utils.misc import prCyan
from carla2gym.carla.util import collision_angle_check, get_ttc

class Graph(object):
    def __init__(self, num_node, max_hop, num_frame, neighbor_dist, neighbor_boundary, sigma, center=-1, weighted_graph=False):
        self.max_hop = max_hop
        self.num_node = num_node
        self.num_frame = num_frame
        self.neighbor_dist = neighbor_dist
        self.neighbor_boundary =  neighbor_boundary
        self.sigma = sigma
        self.center = center if center == -1 else center - 1
        self.num_channel = 10
        self.weighted_graph = weighted_graph

    def _get_adjacency(self, A, weight=None):
        # compute hop steps
        self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0) # arrival mask for each hop

        for d in range(self.max_hop, -1, -1):
            self.hop_dis[arrive_mat[d]] = d

        # compute adjacency
        valid_hop = range(0, self.max_hop + 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            hop_mask = self.hop_dis == hop
            if weight is not None:
                adjacency[hop_mask] = weight[hop_mask]
            else:
                adjacency[hop_mask] = 1
        return adjacency

    def _normalize_adjacency(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)

        valid_hop = range(0, self.max_hop + 1)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]
        return A

    def get_neighbors(self, actors, neighbors = {}):
        self.ego_id = actors[0].id
        ego_trans = actors[0].get_transform()
        ego_loc = ego_trans.location

        filtered_actors = filter(lambda x: x[1] < self.neighbor_boundary,
                                 [(act , act.get_location().distance(ego_loc)) for act in actors if act.is_alive] )
        filtered_sorted_actors = sorted(filtered_actors, key=lambda x: x[1])[:self.num_node]

        # filter out actors who have gone out of bound
        neighbors = dict(filter(lambda it: it[0] in [actor[0].id for actor in filtered_sorted_actors], neighbors.items()))

        # for actor in actors:
        for actor, dist in filtered_sorted_actors:
            if not actor.is_alive:
                continue
            trans = actor.get_transform()
            loc = trans.location

            # global coordinate
            x = loc.x
            y = loc.y
            yaw = np.radians(trans.rotation.yaw)
            vel = actor.get_velocity()
            acc = actor.get_acceleration()
            ang_vel = actor.get_angular_velocity()

            if actor.id not in neighbors:
                neighbors[actor.id] = deque(maxlen=self.num_frame)
                for _ in range(self.num_frame):
                    neighbors[actor.id].append([x, y, yaw, 0., 0., 0., 0., 0., 0., 0.])

            neighbors[actor.id].append([x, y, yaw,
                                        vel.x, vel.y,
                                        acc.x, acc.y,
                                        np.radians(ang_vel.x), np.radians(ang_vel.y),
                                        1.0])

        if len(neighbors.keys()) > self.num_node:
            prCyan("number of neigbor is higher than maximum node")
            filtered_ids = np.array(sorted([(k, np.linalg.norm(v[-1][:2] - np.array([ego_loc.x, ego_loc.y])))
                                            for k, v in neighbors.items()], key=lambda x: x[1]))[:self.self.num_node, 0]
            neighbors = dict(filter(lambda it: it[0] in filtered_ids, neighbors.items()))
            prCyan("after filtering : {}".format(len(neighbors.keys())))

        return neighbors

    def node_n_adacency(self, neighbors):
        cur_ego_features = np.array(neighbors[self.ego_id][self.center])
        ego_yaw = cur_ego_features[2]
        cy = np.cos(ego_yaw)
        sy = np.sin(ego_yaw)
        matrix = np.array([[cy, sy, 0], [sy, -cy, 0], [0, 0, 1]])
        global_to_local = lambda x: np.matmul(matrix, np.vstack([x.T, np.ones((1, x.shape[0]))]))[:2].T

        features = np.zeros([self.num_node, self.num_frame, self.num_channel]) # (V, T, C)
        for i, obj_sess in enumerate(neighbors.values()):
            features[i,...] = obj_sess
            features[i,:,:-1] = features[i,:,:-1] - cur_ego_features[:-1]
            features[i,:,:2] = global_to_local(features[i,:,:2])

        num_obj = np.sum(features[:, self.center, -1]).astype(int)
        xy = features[:num_obj, self.center, :2]
        dist_mat = cdist(xy,xy)
        adj = np.zeros((self.num_node, self.num_node), dtype=np.int64)
        adj[:num_obj, :num_obj] = (dist_mat < self.neighbor_dist).astype(int)
        features = np.transpose(features, (2, 1, 0)) # (C, T, V)

        if self.weighted_graph:
            weight = np.zeros([self.num_node, self.num_node])
            weight[:num_obj, :num_obj] = dist_mat
            weight = np.exp(- weight / self.sigma)
            adj = self._get_adjacency(adj, weight=weight)
        else:
            adj = self._get_adjacency(adj)
        adj = self._normalize_adjacency(adj)

        return features, adj