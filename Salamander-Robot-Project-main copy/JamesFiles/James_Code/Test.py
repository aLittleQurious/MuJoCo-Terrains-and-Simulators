# Import trained model (Exp_1) and try to test


import rospy
import time
import numpy as np
import math
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Twist
import torch

# Import QNet, Brain, and Agent from Exp_1
from Exp_1 import QNet, Brain, Agent

# Global variables to hold robot state
salamander_position_x = 0.0
salamander_position_y = 0.0
salamander_velocity_x = 0.0
salamander_velocity_y = 0.0
spinal_joint_state = 0.0

# Define publishers for joint commands globally
pub_spinal_joint_angle = None

# Initialize the agent
NUM_STATES = 8
NUM_ACTIONS = 3
LEARNING_RATE = 0.0002
DISCOUNT_RATE = 0.99
EPSILON_MIN_VALUE = 0.2
EXPLORATION_DECAY_RATE = 0.999995

agent = Agent(NUM_STATES, NUM_ACTIONS, DISCOUNT_RATE, EXPLORATION_DECAY_RATE, LEARNING_RATE)

# Define reward function
def reward_function(salamander_velocity_x):
    if salamander_velocity_x > 0:
        reward = salamander_velocity_x  # Encourage forward movement
    else:
        reward = -0.1  # Penalize for moving backward or not moving forward
    return reward

def get_link_states_from_environment(data):
    global salamander_position_x, salamander_position_y, salamander_velocity_x, salamander_velocity_y, spinal_joint_state
    indexOf_base_footprint = data.name.index('robot_description::base_footprint')
    salamander_pose = data.pose[indexOf_base_footprint]
    salamander_twist = data.twist[indexOf_base_footprint]
    salamander_position_x = salamander_pose.position.x
    salamander_position_y = salamander_pose.position.y
    salamander_velocity_x = salamander_twist.linear.x
    salamander_velocity_y = salamander_twist.linear.y

    orientation_quaternion = data.pose[indexOf_base_footprint].orientation
    roll, pitch, yaw = euler_from_quaternion([orientation_quaternion.x,
                                              orientation_quaternion.y,
                                              orientation_quaternion.z,
                                              orientation_quaternion.w])
    spinal_joint_state = yaw

def test(episode, agent, is_training=False):
    global salamander_position_x, salamander_position_y, salamander_velocity_x, salamander_velocity_y, spinal_joint_state
    global pub_spinal_joint_angle  # Declare the publisher as global so it is accessible inside this function

    NUM_STEP = 12000  # Set the number of steps for testing
    time_interval = 0.1  # Set time interval for each step

    # Testing environment setup
    goal_position_x = 1
    goal_position_y = 0
    goal_position_diameter = 0.13  # Goal area

    # Initialize previous values
    distance_to_goal = math.sqrt((goal_position_x - salamander_position_x)**2 + abs(goal_position_y - salamander_position_y)**2)
    previous_spinal_joint_angle = spinal_joint_state 
    previous_distance_to_goal = distance_to_goal 

    obs = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype='float64')
    next_obs = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype='float64')

    is_done = False
    episode_reward = 0

    episode_start = time.time()

    # Test loop over steps
    for step in range(NUM_STEP):
        time1 = time.time()

        # Update goal distance
        distance_to_goal_x = goal_position_x - salamander_position_x
        distance_to_goal_y = goal_position_y - salamander_position_y
        distance_to_goal = math.sqrt(distance_to_goal_x**2 + distance_to_goal_y**2)
        angle_to_goal = math.atan2(distance_to_goal_y, distance_to_goal_x)

        # Set up the observation for the next time step
        next_obs[0] = salamander_position_x
        next_obs[1] = salamander_position_y
        next_obs[2] = distance_to_goal
        next_obs[3] = salamander_velocity_x
        next_obs[4] = salamander_velocity_y
        next_obs[5] = angle_to_goal
        next_obs[6] = spinal_joint_state

        # Select action using the agent (no training during testing)
        action = agent.getAction(obs, is_training=False)
        print(f"Selected action: {action}")  # Debug: print selected action
        
        # Check if the goal is reached
        if distance_to_goal <= goal_position_diameter: 
            is_done = True
            reward = 10  # Reward for reaching the goal
            episode_reward += reward
            end_reason = "Goal reached"
            print("Goal reached!")

        # Apply the reward function based on velocity
        reward = reward_function(salamander_velocity_x)
        episode_reward += reward

        # Update the agent (no Q-update during testing)
        obs = np.copy(next_obs)

        # Action-to-joint angle conversion
        # Convert actions to joint angles in the range of [-1.3, 1.3] radians
        action_to_rad = (action - 1) * 0.65  # Action 1 -> 0, Action 0 -> -1.3, Action 2 -> 1.3
        spinal_joint_angle = previous_spinal_joint_angle + action_to_rad

        # Limit spinal joint angle between -1.3 and 1.3 radians
        spinal_joint_angle = max(min(spinal_joint_angle, 1.3), -1.3) 

        # Debugging output
        print(f"Step {step}, Action: {action}, Joint angle: {spinal_joint_angle}")

  
        pub_spinal_joint_angle.publish(spinal_joint_angle)

        # Update previous values for the next loop
        previous_spinal_joint_angle = spinal_joint_angle
        previous_distance_to_goal = distance_to_goal

        # Wait for the simulation to catch up with the control loop
        rospy.sleep(time_interval)  # Sleep to allow Gazebo time to process the movement

   
    print(f"Episode {episode} completed with total reward: {episode_reward}")

def main():
    global pub_spinal_joint_angle  # Declare the publisher as global

    # Initialize the ROS node
    rospy.init_node('salamander_test', anonymous=True)

    # Initialize the ROS publishers
    pub_spinal_joint_angle = rospy.Publisher('/salamander_ns/spinal_controller/command', Float64, queue_size=10)

   
    test(episode=1, agent=agent, is_training=False)

if __name__ == "__main__":
    main()
