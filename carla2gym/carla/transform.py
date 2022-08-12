#!/usr/bin/env python
"""
Tool functions to convert transforms from carla
"""
import carla
import numpy as np
import math

def carla_location_to_XYZ(carla_location):
    """
    Convert a carla location to a x, y, z tuple

    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a tuple with 3 elements (x, y, z)
    :rtype: tuple
    """
    x = carla_location.x
    y = carla_location.y
    z = carla_location.z

    return (x, y, z)

def carla_vector_to_XY(carla_vector):
    """
    Convert a carla vector to a x, y tuple

    :param carla_vector
    :type carla.Vector3D
    :return: a tuple with 2 elements (x, y)
    :rtype: tuple
    """
    x = carla_vector.x
    y = carla_vector.y

    return (x, y)

def carla_location_to_numpy_vector(carla_location):
    """
    Convert a carla location to a numpy array

    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a numpy.array with 3 elements
    :rtype: numpy.array
    """
    x, y, z = carla_location_to_XYZ(carla_location)
    return np.array([x, y, z])

def carla_vector_to_numpy_vector(carla_vector):
    """
    Convert a carla vector to a numpy array

    :param carla_vector
    :type carla.Vector3D
    :return: a numpy.array with 2 elements
    :rtype: numpy.array
    """
    x, y = carla_vector_to_XY(carla_vector)
    return np.array([x, y])

def carla_location_to_delta_vector(carla_location1, carla_location2):
    """
    Calculate a delta location between to carla_location
    """
    location1 = carla_location_to_numpy_vector(carla_location1)
    location2 = carla_location_to_numpy_vector(carla_location2)
    delta_vector = location1 - location2
    return delta_vector

def carla_vectors_to_delta_vector(carla_vector1, carla_vector2):
    """
    Calculate a delta vecotor between two carla_vectors
    """
    vector1 = carla_vector_to_numpy_vector(carla_vector1)
    vector2 = carla_vector_to_numpy_vector(carla_vector2)
    delta_vector = vector1 - vector2
    return delta_vector

def carla_location_euclidean_distance(carla_location_start, carla_location_end):
    """
    Calculate Euclidean distance between two locations
    """
    diff = carla_location_to_delta_vector(carla_location_start, carla_location_end)
    return np.linalg.norm(diff)

def carla_vectors_euclidean_distance(carla_vector1, carla_vector2):
    """
    Calculate Euclidean distance between two vectors
    """
    diff = carla_vectors_to_delta_vector(carla_vector1, carla_vector2)
    return np.linalg.norm(diff)

def carla_location_manhattan_distance(carla_location_start, carla_location_end):
    """
    Calculate manhattan distance between two locations
    """
    diff = carla_location_to_delta_vector(carla_location_start, carla_location_end)
    return np.sum(diff)

def carla_vectors_manhattan_distance(carla_vector1, carla_vector2):
    """
    Calculate manhattan distance between two vectors
    """
    diff = carla_location_to_delta_vector(carla_vector1, carla_vector2)
    return np.sum(diff)

def carla_rotation_to_RPY(carla_rotation):
    """
    Convert a carla rotation to a roll, pitch, yaw tuple

    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a tuple with 3 elements (roll, pitch, yaw)
    :rtype: tuple
    """
    roll = carla_rotation.roll
    pitch = carla_rotation.pitch
    yaw = carla_rotation.yaw

    return (roll, pitch, yaw)

def carla_rotation_to_numpy_vector(carla_rotation):
    """
    Convert a carla rotation to a numpy array

    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a numpy.array with 3 elements
    :rtype: numpy.array
    """
    roll, pitch, yaw = carla_rotation_to_RPY(carla_rotation)
    return np.array([roll, pitch, yaw])

def carla_rotation_to_numpy_array(carla_rotation):
    """
    Convert a carla rotation to a numpy array

    :param carla_rotation: the carla rotation
    :type carla.Rotation
    :return: a numpy.array with 1 element
    :rtype: numpy.array
    """
    return np.array(carla_rotation.yaw)

def carla_rotation_to_delta_angle(carla_rotation1, carla_rotation2):
    """
    Calculate a delta angle between to carla_rotation
    """
    angle1 = carla_rotation_to_numpy_vector(carla_rotation1)
    angle2 = carla_rotation_to_numpy_vector(carla_rotation2)
    delta_angle = angle1 - angle2
    return delta_angle

def carla_rotation_to_delta_yaw(carla_rotation1, carla_rotation2):
    """
    Calculate a delta angle between to carla_rotation
    """
    yaw1 = carla_rotation_to_numpy_array(carla_rotation1)
    yaw2 = carla_rotation_to_numpy_array(carla_rotation2)
    delta_yaw = np.radians(yaw1 - yaw2)
    return delta_yaw

def carla_transform_to_delta_XYZRPY(carla_transform1, carla_transform2):
    loc1 = carla_transform1.location
    rot1 = carla_transform1.rotation
    loc2 = carla_transform2.location
    rot2 = carla_transform2.rotation
    delta_loc = carla_location_to_delta_vector(loc1, loc2)
    delta_angle = carla_rotation_to_delta_angle(rot1, rot2)
    return (*delta_loc, *delta_angle)

def carla_transform_to_delta_XYYAW(carla_transform1, carla_transform2):
    delta_loc = carla_vectors_to_delta_vector(carla_transform1.location, carla_transform2.location)
    delta_yaw = carla_rotation_to_delta_yaw(carla_transform1.rotation, carla_transform2.rotation)
    return (*delta_loc, delta_yaw)

def carla_actor_to_delta_XYZRPYVAW(carla_actor1, carla_actor2):
    """
    return delta location, delta angle, delta velocity, delta accerlation, delta angular velocity
    """
    act1_trans = carla_actor1.get_transform()
    act1_vel = carla_actor1.get_velocity()
    act1_ang_vel = carla_actor1.get_angular_velocity()
    act1_acc = carla_actor1.get_acceleration()

    act2_trans = carla_actor2.get_transform()
    act2_vel = carla_actor2.get_velocity()
    act2_ang_vel = carla_actor2.get_angular_velocity()
    act2_acc = carla_actor2.get_acceleration()

    delta_xyzrpy = carla_transform_to_delta_XYZRPY(act1_trans, act2_trans)
    delta_v = carla_location_to_delta_vector(act1_vel, act2_vel)
    delta_a = carla_location_to_delta_vector(act1_acc, act2_acc)
    delta_w = carla_location_to_delta_vector(act1_ang_vel, act2_ang_vel)

    return (*delta_xyzrpy, *delta_v, *delta_a, *delta_w)

def carla_actor_to_delta_XYYAWVAW(carla_actor1, carla_actor2):
    """
    return delta location, delta angle, delta velocity, delta accerlation, delta angular velocity
    """
    act1_trans = carla_actor1.get_transform()
    act1_vel = carla_actor1.get_velocity()
    act1_ang_vel = carla_actor1.get_angular_velocity()
    act1_acc = carla_actor1.get_acceleration()

    act2_trans = carla_actor2.get_transform()
    act2_vel = carla_actor2.get_velocity()
    act2_ang_vel = carla_actor2.get_angular_velocity()
    act2_acc = carla_actor2.get_acceleration()

    delta_xyyaw = carla_transform_to_delta_XYYAW(act1_trans, act2_trans)
    delta_v = carla_vectors_to_delta_vector(act1_vel, act2_vel)
    delta_a = carla_vectors_to_delta_vector(act1_acc, act2_acc)
    delta_w = np.radians(carla_vectors_to_delta_vector(act1_ang_vel, act2_ang_vel))

    return (*delta_xyyaw, *delta_v, *delta_a, *delta_w)

def _transform_matrix(transform):
    translation = transform.location
    rotation = transform.rotation

    # Transformation matrix
    cy = math.cos(np.radians(rotation.yaw))
    sy = math.sin(np.radians(rotation.yaw))
    cr = math.cos(np.radians(rotation.roll))
    sr = math.sin(np.radians(rotation.roll))
    cp = math.cos(np.radians(rotation.pitch))
    sp = math.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = translation.x
    matrix[1, 3] = translation.y
    matrix[2, 3] = translation.z
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = -cy * sp * cr + sy * sr
    matrix[1, 0] = sy * cp
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = cy * sr - sy * sp * cr
    matrix[2, 0] = sp
    matrix[2, 1] = -cp * sr
    matrix[2, 2] = cp * cr

    return matrix


def transform_points(transform, points):
    """
        Given a 4x4 transformation matrix, transform an array of 3D points.
        Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
        """
    # Needed foramt: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.
    points = points.transpose()
    # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # get transform matrix
    matrix = _transform_matrix(transform)
    # Point transformation
    points = matrix * points
    # Return all but last row
    return points[0:3].transpose()

def local_to_carla_global_tranform_matrix(transform):
    translation = transform.location
    rotation = transform.rotation

    cy = math.cos(np.radians(rotation.yaw))
    sy = math.sin(np.radians(rotation.yaw))

    # Transformation matrix
    matrix = np.identity(4)
    matrix[0, 3] = translation.x
    matrix[1, 3] = translation.y
    matrix[2, 3] = translation.z
    matrix[0, 0] = cy
    matrix[0, 1] = sy
    matrix[1, 0] = sy
    matrix[1, 1] = - cy

    return matrix

def local_to_carla_global_tranform_points(transform, points):
    points = points.transpose()
    # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # get transform matrix
    matrix = local_to_carla_global_tranform_matrix(transform)
    # Point transformation
    points = np.matmul(matrix,points)
    # Return all but last row
    points = points[0:3].transpose()
    return points

def carla_global_to_local_tranform_matrix(transform, rot = True):
    translation = transform.location
    rotation = transform.rotation

    cy = math.cos(np.radians(rotation.yaw))
    sy = math.sin(np.radians(rotation.yaw))
    x = translation.x
    y = translation.y
    z = translation.z

    # Transformation matrix
    matrix = np.identity(4)
    matrix[0, 3] = -x * cy - y * sy
    matrix[1, 3] = -x * sy + y * cy
    matrix[2, 3] = -z
    matrix[0, 0] = cy
    matrix[0, 1] = sy
    matrix[1, 0] = sy
    matrix[1, 1] = -cy


    if rot:
        return rot_90(matrix)
    else:
        return matrix

def rot_90(matrix):
    r_matrix = np.identity(4)
    r_matrix[0, 0] = 0
    r_matrix[0, 1] = -1
    r_matrix[1, 0] = 1
    r_matrix[1, 1] = 0
    return np.matmul(r_matrix, matrix)

def carla_global_v_to_local_v_tranform_matrix(yaw):

    cy = math.cos(np.radians(yaw))
    sy = math.sin(np.radians(yaw))

    # Transformation matrix
    matrix = np.identity(3)
    matrix[0, 0] = cy
    matrix[0, 1] = sy
    matrix[1, 0] = sy
    matrix[1, 1] = -cy

    return matrix

def carla_global_to_local_tranform_points(transform, points, rot=True):
    points = points.transpose()
    # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # get transform matrix
    matrix = carla_global_to_local_tranform_matrix(transform, rot=rot)
    # Point transformation
    points = np.matmul(matrix, points)
    # Return all but last row
    points = points[0:3].transpose()
    return points

def carla_global_to_local_transform_vectors(yaw, vectors):
    vectors = vectors.transpose()
    vectors = np.append(vectors, np.ones((1, vectors.shape[1])), axis=0)
    matrix = carla_global_v_to_local_v_tranform_matrix(yaw)
    vectors = np.matmul(matrix, vectors)
    vectors = vectors[0:2].transpose()
    return vectors
