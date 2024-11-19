import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """
  rot_mat = np.eye(3)
  matrix_k = np.array([[0, axis[2] * -1, axis[1]], 
                      [axis[2], 0, axis[0] * -1],
                      [axis[1] * -1, axis[0], 0]])
  rot_mat += matrix_k * math.sin(angle)
  squared_k = np.matmul(matrix_k, matrix_k)
  rot_mat += squared_k * (1 - math.cos(angle))
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """
  rot_mat = rotation_matrix(axis, angle) 
  T = np.block([[rot_mat, v_A.T], [0,0,0,1]])

  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians

  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """

  hip_frame = homogenous_transformation_matrix([0,0,1], joint_angles[0], np.array([[0,0,0]]))
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  # remove these lines when you write your solution
  # shoulder_frame = np.block(
  #   [[np.eye(3), default_sphere_location.T], 
  #    [0, 0, 0, 1]])
  hip_frame = fk_hip(joint_angles)
  shoulder_frame = homogenous_transformation_matrix([0,1,0], joint_angles[1], np.array([[0,HIP_OFFSET * -1,0]]))
  shoulder_frame = np.matmul(hip_frame, shoulder_frame)
  return shoulder_frame

def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame
  """

  # remove these lines when you write your solution
  # default_sphere_location = np.array([[0.15, 0.1, -0.1]])
  # elbow_frame = np.block(
  #   [[np.eye(3), default_sphere_location.T], 
  #    [0, 0, 0, 1]])
  shoulder_frame = fk_shoulder(joint_angles)
  elbow_frame = homogenous_transformation_matrix([0,1,0], joint_angles[2], np.array([[0,0, UPPER_LEG_OFFSET]]))
  elbow_frame = np.matmul(shoulder_frame, elbow_frame)
  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """

  # remove these lines when you write your solution
  # default_sphere_location = np.array([[0.15, 0.2, -0.1]])
  # end_effector_frame = np.block(
  #   [[np.eye(3), default_sphere_location.T], 
  #    [0, 0, 0, 1]])
  elbow_frame = fk_elbow(joint_angles)
  end_effector_frame = homogenous_transformation_matrix([0,1,0], joint_angles[2], np.array([[0,0, LOWER_LEG_OFFSET]]))
  end_effector_frame = np.matmul(elbow_frame, end_effector_frame)
  return end_effector_frame