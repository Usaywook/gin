from matplotlib.pyplot import axis
import numpy as np

from utils.misc import in_rectangle

import carla

from carla2gym.carla.util import draw_string


def in_elipsoid(pos, coeff):
    return ((pos[:,0] - coeff[2])** 2 / coeff[0]**2) + (pos[:,1] ** 2 / coeff[1] ** 2) <=  1


def get_ttc(obj, vec, coeff):
    def get_x_points(m, o_x, o_y, a, b, c):
        denominator = 1 + m**2
        norminator = m * np.sqrt(b**2 + a**2 / m**2)
        norminator = np.array([norminator, -norminator]) -m*o_y + m**2*o_x + c
        return norminator / denominator

    tangent = vec[1] / vec[0]
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


def collision_angle_check(obj, vec, coeff):
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

def trajectory_hazard(envs, p_traj, mask, args):
    hazards = []
    for e, env in enumerate(envs.venv.envs):
        n_pred = np.transpose(p_traj[e], (2, 1, 0)) # (2, T, V) -> (V, T, 2)
        n_mask = mask[e] # (V)
        num_object = np.sum(n_mask).astype(int)
        traj_ids = env.get_neighbor_keys()
        poly_ids = env.get_polygon_keys()

        collision_ids = []

        traj_box_hazard = [is_traj_box_intersect(traj, env.ego_box) for traj in n_pred[1:num_object]]
        collision_ids.append(traj_ids[np.where(traj_box_hazard)[0] + 1])
        traj_box_hazard = np.any(traj_box_hazard)

        traj_hazard = is_traj_intersect(n_pred[:num_object])
        collision_ids.append(traj_ids[np.where(traj_hazard)[0] + 1])
        traj_hazard = np.any(traj_hazard)

        polygons = env.get_vehicle_polygons()
        ego_poly = polygons[0]
        other_poly  = polygons[1:]
        poly_hazard = [is_polygons_intersect(ego_poly, poly) for poly in other_poly]
        collision_ids.append(poly_ids[np.where(poly_hazard)[0] + 1])
        poly_hazard = np.any(poly_hazard)

        ego_trans = env.actor_info['actors'][0].get_transform()
        ego_loc = ego_trans.location
        ego_yaw = np.radians(ego_trans.rotation.yaw)
        matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw), ego_loc.x],
                    [np.sin(ego_yaw), -np.cos(ego_yaw), ego_loc.y],
                    [0, 0, 1]])
        local_2_global = lambda x: np.matmul(matrix, np.vstack([x.T, np.ones((1, x.shape[0]))]))[:2].T
        ego_traj = local_2_global(n_pred[0])
        traj_poly_hazard = [is_traj_poly_intersect(ego_traj, poly) for poly in other_poly]
        collision_ids.append(poly_ids[np.where(traj_poly_hazard)[0] + 1])
        traj_poly_hazard = np.any(traj_poly_hazard)

        hazard = traj_box_hazard or traj_hazard or traj_poly_hazard or poly_hazard
        hazards.append(hazard)

        collision_ids = np.unique(np.concatenate(collision_ids))
        env.set_hazard(hazard, n_pred[:num_object], traj_ids.tolist(), collision_ids)

    return hazards


def ccw(x1, y1, x2, y2, x3, y3):
    tmp = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return 1 if tmp > 0 else -1 if tmp < 0 else 0

def is_lineseg_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    ans = False
    if ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) == 0 and \
        ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) == 0:
        if min(x1, x2) <= max(x3, x4) and \
            max(x1, x2) >= min(x3, x4) and \
            min(y1, y2) <= max(y3, y4) and \
            min(y3, y4) <= max(y1, y2):

            ans = True
    elif ccw(x1, y1, x2, y2, x3, y3) * ccw(x1, y1, x2, y2, x4, y4) <= 0 and \
        ccw(x3, y3, x4, y4, x1, y1) * ccw(x3, y3, x4, y4, x2, y2) <= 0:
        ans = True
    return ans

def is_traj_intersect(trajectory):
    '''
    trajectory : (V, T, 2)
    '''
    ego_traj = trajectory[0]
    intersects = []
    for other_traj in trajectory[1:]:
        intersect = is_lineseg_intersect(*ego_traj[0], *ego_traj[-1], *other_traj[0], *other_traj[-1])
        intersects.append(intersect)
    return intersects

def is_traj_box_intersect(trajectory, box):
    '''
    trajectory : (V-1, T, 2)
    '''
    intersects = in_rectangle(trajectory.reshape(-1, 2), *box)
    return np.any(intersects)

def is_traj_poly_intersect(trajectory, poly):
    '''
    trajectory : (T, 2)
    poly : (M, 2) - [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
    '''
    sides = np.append(poly, poly[:1], axis=0)
    intersects = []
    for i in range(len(poly)):
        intersect = is_lineseg_intersect(*trajectory[0], *trajectory[-1], *sides[i], *sides[i+1])
        intersects.append(intersect)
    return np.any(intersects)


def is_polygons_intersect(a, b):
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