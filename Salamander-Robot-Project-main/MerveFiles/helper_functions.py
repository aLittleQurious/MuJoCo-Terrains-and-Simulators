import mujoco as mj
import numpy as np
#import gymnasium


def get_joints_velocity(model, data):
    joint_velocities = {}
    for i in range(1,10):
        joint_name = model.joint(i).name
        dof_index = model.jnt_dofadr[i]
        joint_velocities[joint_name] = data.qvel[dof_index]
    
    return joint_velocities
	
	
def get_joints_angle(model, data):
    joint_angles = {}
    for i in range(1,10):
        joint_name = model.joint(i).name
        qpos_index = model.jnt_qposadr[i]
        joint_angles[joint_name] = data.qpos[qpos_index]
    
    return joint_angles
	
	
def get_external_forces(model, data):
    legs = ['leg3_link', 'leg4_link', 'leg1_link', 'leg2_link']
    external_forces = {}
    for i in legs:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, i)
        force = data.cfrc_ext[body_id][:3]
        torque = data.cfrc_ext[body_id][3:]
        external_forces[i] = {"force": force, "torque": torque}
    return external_forces
    
    
def get_goal_angle(a , b):
    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    temp = np.dot(a, b)
    return temp/(a_n * b_n)
    
    
def get_friction_forces(model, data): # we will get values for each leg 
    frictions = {}
    for i in range(data.ncon):
        contact_force = np.zeros(6)
        mj.mj_contactForce(model, data, i, contact_force)
        frictions[f'contact{i}'] = contact_force[:3]
    return frictions
    
    
