#!/usr/bin/env python3

#changed reward function by keeping positive reward if robot moving in positive x direction but adding negative reward otherwise

import rospy
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates              # we can also use ModalStates
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion
from control_msgs.msg import JointControllerState

import math  
from math import degrees

################################################################
LEARNING_RATE = 0.0002   #0.00015  #learning_rate  (This value is better between 0.0001 to 0.01 )
DISCOUNT_RATE = 0.99 #gamma = 0.99
EXPLORATION_DECAY_RATE = 0.999995 #0.9999975   # in Experiment-28 ==> 0.999995   #0.99    #exploration_rate_decay
EPSILON_MIN_VALUE =  0.2  #0.0001       # in Experiment-28 ==> 0.0001   # Initiallt epsilon is 1 and by time it will decrease till this value. 

NUM_EPISODES = 1000
NUM_STEP = 3000 #3000 for healty body and 3500 for broken legs
NUM_ACTIONS = 3   #for  spinal joint
NUM_STATES = 8

ESTIMATED_EPISODE_DURATION = 19  #22   #19.3 
TIME_INTERVAL = 0.006   # Speed for publishing joint commands  =0.009
################################################################
episode_reward_list = []
episode_time_list = []
running_rewards_list=[]
threshold1 = 11.64      # This is the duration we got in active spinal joint without training
min_time=ESTIMATED_EPISODE_DURATION
number_of_beating_episodes = 0


# FOR DATA RECEIVED FROM GAZEBO
salamander_pose = Pose() # we get pose of base_footprint from salamander_robot
salamander_pose = Twist() # we get pose of base_footprint from salamander_robot
salamander_position_x = 0
salamander_position_y = 0
salamander_orientation_z = 0
salamander_velocity_x = 0
salamander_velocity_y = 0


# FOR PUBLISHING DATA
spinal_joint_angle = 0.0
previous_spinal_joint_angle = 0.0
leg1_joint_angle = 0.0
leg2_joint_angle = 0.0
leg3_joint_angle = 0.0
leg4_joint_angle = 0.0
shoulder1_joint_angle = 0.0
shoulder2_joint_angle = 0.0
shoulder3_joint_angle = 0.0
shoulder4_joint_angle = 0.0 
counter=0.0


pub_spinal_joint_angle = rospy.Publisher('/salamander_ns/spinal_controller/command', Float64, queue_size=10)
pub_leg1_joint_angle = rospy.Publisher('/salamander_ns/leg1_controller/command', Float64, queue_size=10)
pub_leg2_joint_angle = rospy.Publisher('/salamander_ns/leg2_controller/command', Float64, queue_size=10)
pub_leg3_joint_angle = rospy.Publisher('/salamander_ns/leg3_controller/command', Float64, queue_size=10)
pub_leg4_joint_angle = rospy.Publisher('/salamander_ns/leg4_controller/command', Float64, queue_size=10)
pub_shoulder1_joint_angle = rospy.Publisher('/salamander_ns/shoulder1_controller/command', Float64, queue_size=10)
pub_shoulder2_joint_angle = rospy.Publisher('/salamander_ns/shoulder2_controller/command', Float64, queue_size=10)
pub_shoulder3_joint_angle = rospy.Publisher('/salamander_ns/shoulder3_controller/command', Float64, queue_size=10)
pub_shoulder4_joint_angle = rospy.Publisher('/salamander_ns/shoulder4_controller/command', Float64, queue_size=10)

class QNet(nn.Module):
	def __init__(self, num_states, dim_mid, num_actions):
		super().__init__()

		self.fc = nn.Sequential(
			nn.Linear(num_states, dim_mid),
			nn.ReLU(),      #relu = rectified linear unit function  Which is activation function
			nn.Linear(dim_mid, dim_mid),
			nn.ReLU(),
			nn.Linear(dim_mid, num_actions)
            #There are three layer fc1 fc2 fc3
            #fc=full connectted layers (some people use dense instead of fc)
            #Here we dont have convolutional layer but if we have it would be like "self.conv1"
		)

	def forward(self, x):       #inplementation of forward pass
		x = self.fc(x)
		return x


class Brain:
    def __init__(self, num_states, num_actions, gamma, r, lr):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = 1.0  
        self.eps_min=EPSILON_MIN_VALUE
        self.gamma = gamma
        self.r = r

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print("self.device = ", self.device)
        ## Q network
        self.q_net = QNet(num_states, 512, num_actions)
        self.q_net.to(self.device)
        ## loss function
        self.criterion = nn.MSELoss()
        ## optimization algorithm
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def updateQnet(self, obs_numpy, action, reward, next_obs_numpy):
        obs_tensor = torch.from_numpy(obs_numpy).float()
        obs_tensor.unsqueeze_(0)	
        obs_tensor = obs_tensor.to(self.device)

        next_obs_tensor = torch.from_numpy(next_obs_numpy).float()
        next_obs_tensor.unsqueeze_(0)
        next_obs_tensor = next_obs_tensor.to(self.device)

        self.optimizer.zero_grad()

        self.q_net.train()
        q = self.q_net(obs_tensor)

        with torch.no_grad():
            self.q_net.eval()	
            label = self.q_net(obs_tensor)
            next_q = self.q_net(next_obs_tensor)

            label[:, action] = reward + self.gamma*np.max(next_q.cpu().detach().numpy(), axis=1)[0]

        loss = self.criterion(q, label)
        loss.backward()
        self.optimizer.step()

    def getAction(self, obs_numpy, is_training):
        if is_training and np.random.rand() < self.eps:
            action = np.random.randint(self.num_actions)
        else:
            obs_tensor = torch.from_numpy(obs_numpy).float()
            obs_tensor.unsqueeze_(0)
            obs_tensor = obs_tensor.to(self.device)
            with torch.no_grad():
                self.q_net.eval()
                q = self.q_net(obs_tensor)
                action = np.argmax(q.cpu().detach().numpy(), axis=1)[0]

        if is_training and self.eps > self.eps_min:
            self.eps *= self.r
        return action

class Agent:
	def __init__(self, num_states, num_actions, gamma, r, lr):
		self.brain = Brain(num_states, num_actions, gamma, r, lr)

	def updateQnet(self, obs, action, reward, next_obs):
		self.brain.updateQnet(obs, action, reward, next_obs)

	def getAction(self, obs, is_training):
		action = self.brain.getAction(obs, is_training)
		return action
      

def get_link_states_from_environment(data):
    global  salamander_position_x,salamander_position_y,salamander_orientation_z,salamander_velocity_x,salamander_velocity_y  

    indexOf_base_footprint = data.name.index('robot_description::base_footprint')  #there are other link also, and message stores them in an array
    salamander_pose=data.pose[indexOf_base_footprint] # Dont forget, pose has 6 data position x,y,z and orientation x,y,z
    salamander_twist=data.twist[indexOf_base_footprint] # Twist has 6 data, we use only velocity_in_X_axis (linear velocity x,y,z and Angular velocity x,y,z)
    salamander_position_x=salamander_pose.position.x 
    salamander_position_y=salamander_pose.position.y
    salamander_velocity_x=salamander_twist.linear.x
    salamander_velocity_y=salamander_twist.linear.y

    orientation_quaternion = data.pose[indexOf_base_footprint].orientation

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion([orientation_quaternion.x,
                                                orientation_quaternion.y,
                                                orientation_quaternion.z,
                                                orientation_quaternion.w])
    salamander_orientation_z =yaw
    #print("x_axis=",salamander_position_x, "      y_axis=",salamander_position_y, "     orientation=",degrees(salamander_orientation_z), "    velocity=",salamander_velocity_x)

def get_spinal_joint_states_from_environment(data):
    global spinal_joint_state
    spinal_joint_state = data.process_value
    #print("Joint spinal_joint_state:", spinal_joint_state)


def reset_simulation():
    rospy.wait_for_service('/gazebo/reset_simulation')
    try:
        reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset_simulation_service()     
        reset_salamander()
        reset_redBall()
        #print("Gazebo simulation reset successful.")  

    except rospy.ServiceException as e:
        print("Service call failed:", str(e))

def reset_salamander():
    rospy.wait_for_service('/gazebo/set_model_state')
    
    model_name = 'robot_description'
    # Replace with the desired initial pose for the model
    initial_pose = {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
    try:
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        model_state_msg = ModelState()
        model_state_msg.model_name = model_name
        model_state_msg.pose.position.x = initial_pose['position']['x']
        model_state_msg.pose.position.y = initial_pose['position']['y']
        model_state_msg.pose.position.z = initial_pose['position']['z']
        model_state_msg.pose.orientation.x = initial_pose['orientation']['x']
        model_state_msg.pose.orientation.y = initial_pose['orientation']['y']
        model_state_msg.pose.orientation.z = initial_pose['orientation']['z']
        model_state_msg.pose.orientation.w = initial_pose['orientation']['w']

        request = SetModelStateRequest()
        request.model_state = model_state_msg

        set_model_state(request)

        pub_spinal_joint_angle.publish(0)
        pub_leg1_joint_angle.publish(0)
        pub_leg2_joint_angle.publish(0)
        pub_leg3_joint_angle.publish(0)
        #pub_leg4_joint_angle.publish(-3.14)   # broken leg
        pub_leg4_joint_angle.publish(0)
        pub_shoulder1_joint_angle.publish(0)
        pub_shoulder2_joint_angle.publish(0)
        pub_shoulder3_joint_angle.publish(0)
        pub_shoulder4_joint_angle.publish(0)
        #print(f"Gazebo model '{model_name}' reset successful.")

    except rospy.ServiceException as e:
        print("Service call failed:", str(e))

def reset_redBall():
    rospy.wait_for_service('/gazebo/set_model_state')
    
    model_name = 'cricket_ball_0'
    # Replace with the desired initial pose for the model
    initial_pose = {'position': {'x': 1.0, 'y': 0.0, 'z': 0.0},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
    try:
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        model_state_msg = ModelState()
        model_state_msg.model_name = model_name
        model_state_msg.pose.position.x = initial_pose['position']['x']
        model_state_msg.pose.position.y = initial_pose['position']['y']
        model_state_msg.pose.position.z = initial_pose['position']['z']
        model_state_msg.pose.orientation.x = initial_pose['orientation']['x']
        model_state_msg.pose.orientation.y = initial_pose['orientation']['y']
        model_state_msg.pose.orientation.z = initial_pose['orientation']['z']
        model_state_msg.pose.orientation.w = initial_pose['orientation']['w']

        request = SetModelStateRequest()
        request.model_state = model_state_msg

        set_model_state(request)

    except rospy.ServiceException as e:
        print("Service call failed:", str(e))
    
def reward_function(salamander_velocity_x):
    # Constants for reward shaping

    # Calculate the reward
    if salamander_velocity_x > 0:
        reward = salamander_velocity_x  # Encourage forward movement
    else:
        reward = -0.1  # Penalize for moving backward or not moving forward
    
    return reward
      

def simulate(episode, is_training):
    global  salamander_position_x,salamander_position_y,salamander_orientation_z,salamander_velocity_x,salamander_velocity_y, spinal_joint_state
    
    goal_position_x = 1
    goal_position_y = 0
    goal_position_diameter = 0.13    
    distance_to_goal_x= goal_position_x-salamander_position_x
    distance_to_goal_y= goal_position_y-salamander_position_y
    distance_to_goal=math.sqrt( (goal_position_x-salamander_position_x)**2 + abs(goal_position_y-salamander_position_y)**2 )  #Euclidean Distance
    previous_position_x=salamander_position_x
    previous_position_y=salamander_position_x
    previous_spinal_joint_angle = spinal_joint_state    
    previous_distance_to_goal_x = distance_to_goal_x
    previous_distance_to_goal_y = distance_to_goal_y
    previous_distance_to_goal =distance_to_goal  #Initially

    obs = np.array([0,0,0,0,0,0,0,0], dtype='float64')
    next_obs = np.array([0,0,0,0,0,0,0,0], dtype='float64')
    max_step = NUM_STEP     #3000   #Hardcoded takes 1550 steps in avarage to reach goal at time_interval = 0.006
    estimated_episode_duration=ESTIMATED_EPISODE_DURATION   #For 3000 steps it takes around 18... seconds in this pc.

    is_done = False
    time_interval = TIME_INTERVAL
    episode_reward = 0 

    counter=0       # reset Hildebrand Gait Sequence  

    episode_start = time.time()          

    csvFile = open("gaits.csv")
    gaits = np.loadtxt(csvFile, delimiter=",")

    for step in range(max_step):
        time1 = time.time()

        distance_to_goal_x= goal_position_x-salamander_position_x
        distance_to_goal_y= goal_position_y-salamander_position_y
        distance_to_goal= math.sqrt( (goal_position_x-salamander_position_x)**2 + abs(goal_position_y-salamander_position_y)**2 )  #Euclidean Distance 
        angle_to_goal = math.atan2(goal_position_y - salamander_position_y, goal_position_x - salamander_position_x)
        delta_distance_to_goal_x = previous_distance_to_goal_x - distance_to_goal_x # Topu gecene kadar position, topu gectikten sonra negative (ve yaklastikca +ve deger)
        delta_distance_to_goal_y = previous_distance_to_goal_y - distance_to_goal_y  #Sol tarafa giderse negative, sag tarafa giderse positive
        delta_distance_to_goal= previous_distance_to_goal - distance_to_goal #Gaol'e dogru yuruyunce artar, oteki turlu azalir?
        elapsed_time = step * time_interval
        #print(delta_distance_to_goal*10)

        # Robot duz cizgiye yakin yurudugunde delta_distance_to_goal ve delta_distance_to_goal_x biribirine cok yakin degerler ve delta_distance_to_goal_y bu ikisinden cok kucuk kaliyor 
        #print(delta_distance_to_goal*1000 ,delta_distance_to_goal_x*1000 , delta_distance_to_goal_y*1000) 
                
        next_obs[0] = salamander_position_x
        next_obs[1] = salamander_position_y
        next_obs[2] = distance_to_goal
        next_obs[3] = salamander_velocity_x
        next_obs[4] = salamander_velocity_y
        next_obs[5] = angle_to_goal
        next_obs[6] = spinal_joint_state
        next_obs[7] = elapsed_time  

        if(step == 0):
            next_obs[0] = 0
            next_obs[1] = 0
            next_obs[2] = 0
            next_obs[3] = 0
            next_obs[4] = 0
            next_obs[5] = 0
            next_obs[6] = 0
            next_obs[7] = 0

        action = agent.getAction(obs, is_training)     

        
        if step == max_step-1 or distance_to_goal<=goal_position_diameter : 
            is_done = True                 

        #REWARD FUNCTION  
        reward = reward_function(salamander_velocity_x)
        episode_reward += reward

        resolution = 1      # If robot can't chatch that angle in time then increase this value
        action_to_rad = (action-1) / resolution  # num_actions=21 from 0 to 2 , means here we convert  action to -1 to 1
        action_to_rad= action_to_rad*0.0174533  # 1-degree is 0.0174533rad  
        #print(action_to_rad)

        if is_training:
            agent.updateQnet(obs, action, reward, next_obs)

        obs = np.copy(next_obs)

        #spinal_joint_angle=0
        #spinal_joint_angle=gaits[0][counter]
        spinal_joint_angle= previous_spinal_joint_angle + action_to_rad         # incremenatal angle change
        #print(spinal_joint_angle)          
        if spinal_joint_angle>1.3:    # 0.52355109694411 for Hildebrand and 1.35 from URDF file
              spinal_joint_angle = 1.3
        if spinal_joint_angle<-1.3:
              spinal_joint_angle = -1.3      
        leg1_joint_angle=gaits[1][counter]
        leg2_joint_angle=gaits[2][counter]
        leg3_joint_angle=gaits[3][counter]
        leg4_joint_angle=gaits[4][counter]
        #leg4_joint_angle= -3.14                     # For broken leg
        shoulder1_joint_angle=gaits[5][counter]
        shoulder2_joint_angle=gaits[6][counter]
        shoulder3_joint_angle=gaits[7][counter]
        shoulder4_joint_angle=gaits[8][counter]
        
        pub_spinal_joint_angle.publish(spinal_joint_angle)
        pub_leg1_joint_angle.publish(leg1_joint_angle)
        pub_leg2_joint_angle.publish(leg2_joint_angle)
        pub_leg3_joint_angle.publish(leg3_joint_angle)
        pub_leg4_joint_angle.publish(leg4_joint_angle)
        pub_shoulder1_joint_angle.publish(shoulder1_joint_angle)
        pub_shoulder2_joint_angle.publish(shoulder2_joint_angle)
        pub_shoulder3_joint_angle.publish(shoulder3_joint_angle)
        pub_shoulder4_joint_angle.publish(shoulder4_joint_angle)

        # UPDATE COUNTER FOR READING NEW COLOMNS FROM .csv file
        if counter == 247:               #gait squence size in csv file
            counter=0
        else:
            counter=counter+1

        #Update previous values
        previous_position_x=salamander_position_x
        previous_position_y=salamander_position_y
        previous_spinal_joint_angle = spinal_joint_angle  
        previous_distance_to_goal = distance_to_goal
        previous_distance_to_goal_x = distance_to_goal_x
        previous_distance_to_goal_y = distance_to_goal_y
   
        time2 = time.time()
        interval = time2 - time1
        if(interval < time_interval):
            time.sleep(time_interval - interval)

        if (is_done and is_training):
            episode_end = time.time()  
            episode_time = episode_end-episode_start 
            print('Episode: {0} Finished after {1}sn  steps {2} with reward {3}'.format(episode+1, episode_time, step+1, episode_reward))
            #print('Time={0}     Reward{1}'.format(episode_time, episode_reward))
            plot_durations(episode_reward,episode_time)
            break
        elif(is_done and is_training == False):
            print('Evaluation: Finished after {} time steps'.format(step+1))
            break

def plot_durations(episode_reward, episode_time):
    global min_time
    plt.figure("Training")
    plt.clf()
    plt.suptitle("Training is in PROCESS")

    if episode_time < min_time:
        min_time=episode_time    

    # Plot 1: Rewards in Episodes
    episode_reward_list.append(episode_reward)
    mean_of_last_rewards = np.mean(episode_reward_list)
    running_rewards_list.append(mean_of_last_rewards)
    x1 = np.arange(0, len(episode_reward_list))
    plt.subplot(1, 2, 1)
    plt.title("Rewards in Episodes")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    #plt.plot(x1, episode_reward_list, ".")
    plt.plot(x1, episode_reward_list,".", label='actual reward')
    plt.plot(x1, running_rewards_list, label='running reward')
    #plt.ylim(-500, 20000)

    # Plot 2: Time to reach goal
    episode_time_list.append(episode_time)
    x2 = np.arange(0, len(episode_time_list))
    plt.subplot(1, 2, 2)
    plt.title("Durations in Episodes")
    plt.xlabel('Episode')
    plt.ylabel('Duration (sn)')
    plt.plot(x2, episode_time_list, ".")
    #plt.plot(x2, episode_time_list)
    plt.ylim(0, 50)  # If episode_time is bigger than 100 then change it otherwise it will crop +100 values 
    #plt.axhline(y=threshold1, color='r', linestyle='--', label='Threshold')
    #plt.text(-0.5, threshold1, f'Threshold: {threshold1}', color='r', fontsize=7, ha='left')

    plt.axhline(y=min_time, color='g', linestyle='--', label='min_time')
    plt.text(-0.5, min_time, f'min_time: {min_time}', color='g', fontsize=7, ha='left')


    plt.pause(0.001)


def save_model(model, filename='/home/minirolab/Desktop/Salamander/ROS-WorkSpace/src/salamander_development/Trained_Model/trained_qnet.pth'):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the model's state_dict
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

if __name__ == '__main__':
    rospy.init_node('Salamander_simulation', anonymous=True)
    rospy.Subscriber("/gazebo/link_states", LinkStates, get_link_states_from_environment)
    rospy.Subscriber('/salamander_ns/spinal_controller/state', JointControllerState, get_spinal_joint_states_from_environment)

    print("NUM_STATES",NUM_STATES)
    print("NUM_ACTIONS",NUM_ACTIONS)
    print("DISCOUNT_RATE",DISCOUNT_RATE)
    print("EXPLORATION_DECAY_RATE",EXPLORATION_DECAY_RATE)
    print("EPSILON_MIN_VALUE",EPSILON_MIN_VALUE)
    print("LEARNING_RATE",LEARNING_RATE)    
    print("NUM_EPISODES",NUM_EPISODES)
    print("NUM_STEP",NUM_STEP)
    print("ESTIMATED_EPISODE_DURATION",ESTIMATED_EPISODE_DURATION)
    print("TIME_INTERVAL",TIME_INTERVAL)


    agent = Agent(NUM_STATES, NUM_ACTIONS, DISCOUNT_RATE, EXPLORATION_DECAY_RATE, LEARNING_RATE)

    num_episodes = NUM_EPISODES
    is_training = True
    training_start_time = time.time()
    
    for i_episode in range(num_episodes):
        reset_simulation()
        time.sleep(0.1)
        simulate(i_episode, is_training)

    save_model(agent.brain.q_net)
    print('Model Saved')
    
    training_end_time = time.time()
    training_duration_seconds=training_end_time-training_start_time
    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(training_duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Training is DONE in = {} hours, {} minutes, and {:.2f} seconds'.format(int(hours), int(minutes), seconds))

    # RECORD CSV FILE
    save_time_data_in_cvs = 'episode_times.csv'
    # Write the array to a CSV file
    np.savetxt(save_time_data_in_cvs, episode_time_list, delimiter=',')

    #PLOT GRAPH
    plt.suptitle('Training is DONE in = {} hours, {} minutes, and {:.2f} seconds'.format(int(hours), int(minutes), seconds))
    plt.show()

    

