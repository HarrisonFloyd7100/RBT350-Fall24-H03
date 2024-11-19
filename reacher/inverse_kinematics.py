import math
import numpy as np
import copy
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 10
DELTA = 0.0001

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    # Initialize cost to zero
    cost = 0.0

    # Add your solution here.
    forward_position = forward_kinematics.fk_foot(guess)[0:3,3]
    cost = np.linalg.norm(end_effector_pos - forward_position)
    return cost


def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """
    joint_angles = np.transpose(joint_angles)
    # Initialize Jacobian to zero
    J = np.zeros([3, 3])

    # Add your solution here.
    for i in range(3):
        for j in range(3):
            temp_joint_angles = joint_angles
            starting_pos = forward_kinematics.fk_foot(temp_joint_angles) # finds the current position
            temp_joint_angles[j] += delta # applies the small change
            change_in_pos = forward_kinematics.fk_foot(temp_joint_angles) # calculates the new position

        
            J[i][j] = ( (change_in_pos[i][3] - starting_pos[i][3] ) / delta ) # calculates the d/dq

    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """

    # Initialize previous cost to infinity
    previous_cost = np.inf
    # Initialize the current cost to 0.0
    cost = 0.0

    q_current = guess # initializes the current angles
    q_next = np.zeros(3) # initializes next angles

    for iters in range(MAX_ITERATIONS):

        # Calculate the Jacobian matrix using finite differences
        J = calculate_jacobian_FD(q_current,DELTA)

        fks = forward_kinematics.fk_foot(np.transpose(q_current))
        a = [[fks[0][3]], [fks[1][3]], [fks[2][3]]]
        foot_pos = np.transpose(np.array(a))
        # Calculate the distance from our target for each position(x,y,z) 
        distance_from_target = np.subtract(end_effector_pos, foot_pos) # distance = target - f(q)
        print("distance from target is:\n{}".format(distance_from_target))
        # Compute the step to update the joint angles using the Moore-Penrose pseudoinverse using numpy.linalg.pinv
        J_inv = np.linalg.pinv(J)


        # Take a full Newton step to update the guess for joint angles
        q_next = q_current + np.transpose(np.matmul(J_inv,np.transpose(distance_from_target)))
        q_current = q_next
        # finds the next position we would take with this q_next
        fks = forward_kinematics.fk_foot(np.transpose(q_next))
        a = [[fks[0][3]], [fks[1][3]], [fks[2][3]]]
        foot_pos = np.array(a)

        

        print("the desired position is: \n {}".format(end_effector_pos))
        print("the current position is: \n {}".format(foot_pos))
        print("current q: {}".format(q_current))
        cost = ik_cost(end_effector_pos, foot_pos) #calculates cost of that decision
        # Calculate the cost based on the updated guess
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost
        print("the cost is: {}".format(cost))

    
    return np.transpose(q_next)
