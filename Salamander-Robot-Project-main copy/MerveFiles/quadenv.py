# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/mujoco_env.py
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v5.py

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv

import numpy as np
#from gymnasium.envs import mujoco
import mujoco
#import glfw
import helper_functions as hf

from gymnasium import utils
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}

# If You Want to Omit Spinal Joint, Uncomment Lines 102-4 in Azhang.xml file. It will act like there is no joint there.

class MyQuadRobotEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
           # "rgbd_tuple",
        ],
    }
    
    

    def __init__(self, 
                 hildebrand_enabled=False,
                 spinal_joint_enabled=False, 
                 goal_enabled=False,
                 xml_file="/home1/atasever/Robotics/Salamander_Robot/Salamander-Robot-Project/xml_files/azhang.xml", 
                 render_mode=None,
                 frame_skip=5.0,
                 include_cfrc_ext_in_observation=False,
                 forward_reward_weight= 1.0,
                 ctrl_cost_weight= 0.5,
                 contact_cost_weight=5e-4,
                 goal_reward_weight=1.0,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.07, 0.09),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 #main_body=1,   # Name or ID of the body, whose displacement is used to calculate forward_reward
                 #default_camera_config=DEFAULT_CAMERA_CONFIG,
                 **kwargs):
                 
        utils.EzPickle.__init__(self, 
                                 hildebrand_enabled,
                                 spinal_joint_enabled, 
                                 goal_enabled,
                                 xml_file,
                                 render_mode,
                                 frame_skip,
                                 include_cfrc_ext_in_observation,
                                 forward_reward_weight,
                                 ctrl_cost_weight,
                                 contact_cost_weight,
                                 goal_reward_weight,
                                 healthy_reward,
                                 terminate_when_unhealthy,
                                 healthy_z_range,
                                 contact_force_range,
                                 reset_noise_scale,
                                 exclude_current_positions_from_observation,
                                 #main_body,
                                 #default_camera_config,
                                 **kwargs)
        
        self._hildebrand_enabled = hildebrand_enabled
        self._spinal_joint_enabled = spinal_joint_enabled
        self._goal_enabled = goal_enabled
        
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._goal_reward_weight = goal_reward_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        #self._main_body = main_body
        
        #self.render_mode = render_mode

        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)
        
        MujocoEnv.__init__(self,
                            xml_file,
                            render_mode,
                            frame_skip,
                            #observation_space=None,  
                            #default_camera_config=default_camera_config,
                            **kwargs,)


        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
               # "rgbd_tuple",
            ],
            "render_fps": 50,
        }


        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.torso_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "torso_sensors")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body_rear_link")
        
        self._main_body = self.torso_site_id
        
        obs_dim = 0 # obs_shape
        if self._spinal_joint_enabled:
            if self._hildebrand_enabled:
                self.initialize_hildebrand()
                if self._goal_enabled:
                    self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                    self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_sensors")
                    
                    obs_dim = 3 + 4 + 9 + 3 + 3 + 9 + 12 + 2   # if states doubles, it should have been  6 + 8 + (16+1) + 6 + 6 + (16+1) + 12 + 2  # 45
                else:
                    
                    obs_dim = 3 + 4 + 9 + 3 + 3 + 9 + 12  # 43
                    
                self.action_space = Box(low=-1.57, high=1.57, shape=(1,), dtype=np.float64)   
            else:
                if self._goal_enabled:
                    ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                    tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_sensors")
                    
                    obs_dim = 3 + 4 + 9 + 3 + 3 + 9 + 12 + 2  # 45
                else:
                    
                    obs_dim = 3 + 4 + 9 + 3 + 3 + 9 + 12  # 43
                    
                self.action_space = Box(low=-1.57, high=1.57, shape=(9,), dtype=np.float64)   
                
        else: # no hildebrand, no spinal joint
            if self._goal_enabled:
                ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_sensors")
                
                obs_dim = 3 + 4 + 8 + 3 + 3 + 8 + 12 + 2  # 43    
            
            else:
                
                obs_dim = 3 + 4 + 8 + 3 + 3 + 8     # + 12   41-12 for now
                
            self.action_space = Box(low=-1.57, high=1.57, shape=(8,), dtype=np.float64)   
         
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        #Reset the simulation state
        self.reset()
        
        
    def initialize_hildebrand(self):
        
        csvFile = open("gaits.csv")
        gaits = np.loadtxt(csvFile, delimiter=",")
        self.seq = 0

        #spinal_joint_angle=gaits[0]
        self.leg1_joint_angle=gaits[1]
        self.leg2_joint_angle=gaits[2]
        self.leg3_joint_angle=gaits[3]
        self.leg4_joint_angle=gaits[4]
        self.shoulder1_joint_angle=gaits[5]
        self.shoulder2_joint_angle=gaits[6]
        self.shoulder3_joint_angle=gaits[7]
        self.shoulder4_joint_angle=gaits[8]

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy
        
    def _get_obs(self):
    
        if self._spinal_joint_enabled:
            if self._hildebrand_enabled:
                if self._goal_enabled:
                    
                    states_C = self.data.site_xpos[self.torso_site_id]  # coordinates of torso
                    states_D = self.data.xquat[self.torso_id]   # orientations of torso
                    states_A = hf.get_joints_angle(self.model, self.data)
                    states_F = self.data.sensordata[7:10]  # torso_angular_velocity
                    states_E = self.data.sensordata[10:]    # torso_linear_velocity
                    states_B = hf.get_joints_velocity(self.model, self.data) 
                    #states_G = hf.get_external_forces(self.model, self.data)
                    states_R = hf.get_friction_forces(self.model, self.data)
                    
                    ball_pos = self.data.xpos[self.ball_id]
                    tip_pos = self.data.site_xpos[self.tip_site_id]
                    states_T = ball_pos - tip_pos
                    states_Ts = hf.get_goal_angle(ball_pos, tip_pos)
                    
                    
                    states_A = np.array(list(states_A.values()))
                    states_B = np.array(list(states_B.values()))
                    states_R = np.array(list(states_R.values())).flatten()
                    states_Ts = np.array(states_Ts).flatten()
                    
                    obs = np.concatenate((states_C, states_D, states_A, states_F, states_E, states_B, states_R, states_T, states_Ts))
                else:
                    states_C = self.data.site_xpos[self.torso_site_id]  # coordinates of torso
                    states_D = self.data.xquat[self.torso_id]   # orientations of torso
                    states_A = hf.get_joints_angle(self.model, self.data)
                    states_F = self.data.sensordata[7:10]  # torso_angular_velocity
                    states_E = self.data.sensordata[10:]    # torso_linear_velocity
                    states_B = hf.get_joints_velocity(self.model, self.data) 
                    #states_G = hf.get_external_forces(self.model, self.data)
                    states_R = hf.get_friction_forces(self.model, self.data)
                    
                    
                    states_A = np.array(list(states_A.values()))
                    states_B = np.array(list(states_B.values()))
                    states_R = np.array(list(states_R.values())).flatten()
                    
                    obs = np.concatenate((states_C, states_D, states_A, states_F, states_E, states_B, states_R))
                
            else:
                if self._goal_enabled:
                    states_C = self.data.site_xpos[self.torso_site_id]  # coordinates of torso
                    states_D = self.data.xquat[self.torso_id]   # orientations of torso
                    states_A = hf.get_joints_angle(self.model, self.data)
                    states_F = self.data.sensordata[7:10]  # torso_angular_velocity
                    states_E = self.data.sensordata[10:]    # torso_linear_velocity
                    states_B = hf.get_joints_velocity(self.model, self.data) 
                    #states_G = hf.get_external_forces(self.model, self.data)
                    states_R = hf.get_friction_forces(self.model, self.data)
                    
                    ball_pos = self.data.xpos[self.ball_id]
                    tip_pos = self.data.site_xpos[self.tip_site_id]
                    states_T = ball_pos - tip_pos
                    states_Ts = hf.get_goal_angle(ball_pos, tip_pos)
                    
                    
                    states_A = np.array(list(states_A.values()))
                    states_B = np.array(list(states_B.values()))
                    states_R = np.array(list(states_R.values())).flatten()
                    states_Ts = np.array(states_Ts).flatten()
                    
                    obs = np.concatenate((states_C, states_D, states_A, states_F, states_E, states_B, states_R, states_T, states_Ts))
                else:
                    states_C = self.data.site_xpos[self.torso_site_id]  # coordinates of torso
                    states_D = self.data.xquat[self.torso_id]   # orientations of torso
                    states_A = hf.get_joints_angle(self.model, self.data)
                    states_F = self.data.sensordata[7:10]  # torso_angular_velocity
                    states_E = self.data.sensordata[10:]    # torso_linear_velocity
                    states_B = hf.get_joints_velocity(self.model, self.data) 
                    #states_G = hf.get_external_forces(self.model, self.data)
                    states_R = hf.get_friction_forces(self.model, self.data)
                    
                    
                    states_A = np.array(list(states_A.values()))
                    states_B = np.array(list(states_B.values()))
                    states_R = np.array(list(states_R.values())).flatten()
                    
                    obs = np.concatenate((states_C, states_D, states_A, states_F, states_E, states_B, states_R))
                    
        else: # no hildebrand, no spinal joint
            if self._goal_enabled:
                states_C = self.data.site_xpos[self.torso_site_id]  # coordinates of torso
                states_D = self.data.xquat[self.torso_id]   # orientations of torso
                states_A = hf.get_joints_angle(self.model, self.data)
                states_F = self.data.sensordata[7:10]  # torso_angular_velocity
                states_E = self.data.sensordata[10:]    # torso_linear_velocity
                states_B = hf.get_joints_velocity(self.model, self.data) 
                #states_G = hf.get_external_forces(self.model, self.data)
                states_R = hf.get_friction_forces(self.model, self.data)
                
                ball_pos = self.data.xpos[self.ball_id]
                tip_pos = self.data.site_xpos[self.tip_site_id]
                states_T = ball_pos - tip_pos
                states_Ts = hf.get_goal_angle(ball_pos, tip_pos)
                
                
                states_A = np.array(list(states_A.values()))
                states_B = np.array(list(states_B.values()))
                states_R = np.array(list(states_R.values())).flatten()
                states_Ts = np.array(states_Ts).flatten()
                
                obs = np.concatenate((states_C, states_D, states_A, states_F, states_E, states_B, states_R, states_T, states_Ts)) 
            
            else:
                states_C = self.data.site_xpos[self.torso_site_id]  # coordinates of torso
                states_D = self.data.xquat[self.torso_id]   # orientations of torso
                states_A = hf.get_joints_angle(self.model, self.data)
                states_F = self.data.sensordata[7:10]  # torso_angular_velocity
                states_E = self.data.sensordata[10:]    # torso_linear_velocity
                states_B = hf.get_joints_velocity(self.model, self.data) 
                #states_G = hf.get_external_forces(self.model, self.data)
                #states_R = hf.get_friction_forces(self.model, self.data)
                
                
                states_A = np.array(list(states_A.values()))
                states_B = np.array(list(states_B.values()))
                #states_R = np.array(list(states_R.values())).flatten()
                
                print("states_C size:", states_C.shape)  # 3
                print("states_D size:", states_D.shape)  # 4
                print("states_A size:", states_A.shape)  # 9
                print("states_F size:", states_F.shape)  # 3
                print("states_E size:", states_E.shape)  # 3
                print("states_B size:", states_B.shape)  # 9 
                
                obs = np.concatenate((states_C, states_D, states_A, states_F, states_E, states_B)) # states_R
            
        return obs

    def step(self, action):  
    
        x_position_before = self.data.body(self._main_body).xpos[0].copy()
        
        if self._goal_enabled:
            ball_pos = self.data.xpos[self.ball_id]
            tip_pos = self.data.site_xpos[self.tip_site_id]
            dist_to_goal_before = ball_pos - tip_pos
        else:
            dist_to_goal_before = 0.0

        if self._hildebrand_enabled:
            self.data.qpos[7 + 6] = self.leg1_joint_angle[self.seq]
            self.data.qpos[9 + 6] = self.leg2_joint_angle[self.seq]
            self.data.qpos[2 + 6] = self.leg3_joint_angle[self.seq]
            self.data.qpos[4 + 6] = self.leg4_joint_angle[self.seq]
            self.data.qpos[6 + 6] = self.shoulder1_joint_angle[self.seq]
            self.data.qpos[8 + 6] = self.shoulder2_joint_angle[self.seq]
            self.data.qpos[1 + 6] = self.shoulder3_joint_angle[self.seq]
            self.data.qpos[3 + 6] = self.shoulder4_joint_angle[self.seq]
            self.data.ctrl[:] = action   # self.data.ctrl[0] = action
            
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        
            self.seq +=1
        else:
            self.data.ctrl[:] = action 
            
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
            
        #mujoco.mj_rnePostConstraint(self.model, self.data)
        
        x_position_after = self.data.body(self._main_body).xpos[0].copy()
        x_velocity = (x_position_after - x_position_before) / self.dt
        #x_velocity = self.data.qvel[0]
        
        if self._goal_enabled:
            ball_pos = self.data.xpos[self.ball_id]
            tip_pos = self.data.site_xpos[self.tip_site_id]
            dist_to_goal_after = ball_pos - tip_pos
            dist_change = dist_to_goal_before - dist_to_goal_after
        else:
            dist_change = 0.0
        
        obs = self._get_obs()  

        reward = self._get_rew(x_velocity, dist_change, action)
        done = (not self.is_healthy) and self._terminate_when_unhealthy

        info = {}
        
        if self.render_mode == "human":
            self.render()
        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`        
        return obs, reward, done, False, info 
        
        
    def _get_rew(self, x_velocity: float, dist_change: float, action):
    
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = 0 #self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
    
        if self._goal_enabled:
            goal_reward = dist_change * self._goal_reward_weight
            reward += goal_reward

        return reward


    def reset_model(self):
        
        # Set initial state if needed, for example a slight perturbation:
        #self.data.qpos[:] = 0.0
        #self.data.qvel[:] = 0.0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = (self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv))
        
        self.set_state(qpos, qvel)

        return self._get_obs()
        

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

