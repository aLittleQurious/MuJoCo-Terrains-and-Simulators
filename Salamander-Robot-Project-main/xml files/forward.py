import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys

xml_path = 'azhang.xml' #xml file (assumes this is in the same folder as this file)
simend = 100 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

csvFile = open("gaits.csv")
gaits = np.loadtxt(csvFile, delimiter=",")
seq = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

def get_joints_velocity(model, data):
    joint_velocities = {}
    flag = False
    if model.neq > 0:
        flag = True
    for i in range(1,10):
        if flag and (i==5):
            continue
        joint_name = model.joint(i).name
        dof_index = model.jnt_dofadr[i]
        joint_velocities[joint_name] = data.qvel[dof_index]
    
    return joint_velocities

def get_joints_angle(model, data):
    joint_angles = {}
    flag = False
    if model.neq > 0:
        flag = True
    for i in range(1,10):
        if flag and (i==5):
            continue
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

def get_friction_forces(model, data):
    frictions = {'leg1_link':np.zeros(3),
                 'leg2_link':np.zeros(3),
                 'leg3_link':np.zeros(3),
                 'leg4_link':np.zeros(3)}
    for i in range(data.ncon):
        contact = data.contact[i]
        geom2_id = contact.geom2
        body2_id = model.geom_bodyid[geom2_id]
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body2_id)
        contact_force = np.zeros(6)
        mj.mj_contactForce(model, data, i, contact_force)
        frictions[body] = contact_force[:3]
    return frictions

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
#initialize the controller here. This function is called once, in the beginning
cam.azimuth = 90
cam.elevation = -45
cam.distance = 3
cam.lookat = np.array([1.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

N = 247
spinal_joint_angle=gaits[0]
leg1_joint_angle=gaits[1]
leg2_joint_angle=gaits[2]
leg3_joint_angle=gaits[3]
leg4_joint_angle=gaits[4]
shoulder1_joint_angle=gaits[5]
shoulder2_joint_angle=gaits[6]
shoulder3_joint_angle=gaits[7]
shoulder4_joint_angle=gaits[8]

#initialize
data.qpos[7] = 0
data.qpos[9] = 0
data.qpos[2] = 0
data.qpos[4] = 0
data.qpos[6] = 0
data.qpos[8] = 0
data.qpos[1] = 0
data.qpos[3] = 0
time = 0
dt = 0.003
ball_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ball")
torso_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "body_rear_link")
torso_joint_id = model.body_jntadr[torso_id]
tip_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "tip_sensors")
torso_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "torso_sensors")
sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "velocity")
sensor_id2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "spinal_quat")
spinal_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "spinal_joint")

for joint_id in range(model.njnt):
    joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
    qpos_address = model.jnt_qposadr[joint_id]
    qvel_address = model.jnt_dofadr[joint_id]
    joint_type = model.jnt_type[joint_id]  # 0: free, 1: ball, 2: slide, 3: hinge
    print(f"Joint {joint_id} ({joint_name}): QPOS Address {qpos_address}, QVEL Address {qvel_address} Joint Type {joint_type}")
print(len(data.sensordata))
print(f'limits on spinal joint are: {model.jnt_range[spinal_joint_id]}')

while not glfw.window_should_close(window):
    time_prev = time

    while (time - time_prev < 1.0/300.0):
        data.qpos[7 + 6] = leg1_joint_angle[seq]
        data.qpos[9 + 6] = leg2_joint_angle[seq]
        data.qpos[2 + 6] = leg3_joint_angle[seq]
        data.qpos[4 + 6] = leg4_joint_angle[seq]
        data.qpos[6 + 6] = shoulder1_joint_angle[seq]
        data.qpos[8 + 6] = shoulder2_joint_angle[seq]
        data.qpos[1 + 6] = shoulder3_joint_angle[seq]
        data.qpos[3 + 6] = shoulder4_joint_angle[seq]
        data.ctrl[0] = spinal_joint_angle[seq]
        if seq == 247:
            seq = 0
        else:
            seq +=1
        #site_pos = data.site_xpos[site_id]
        start_vel_index = model.jnt_dofadr[torso_joint_id]
        state_B = get_joints_velocity(model, data)
        state_A = get_joints_angle(model, data)
        state_C = data.site_xpos[torso_site_id]
        state_D = data.xquat[torso_id]
        ball_pos = data.xpos[ball_id]
        tip_pos = data.site_xpos[tip_site_id]
        state_T = ball_pos - tip_pos
        state_Ts = get_goal_angle(ball_pos, tip_pos)
        torso_angular_velocity = data.sensordata[7:10]  #state_F
        torso_linear_velocity = data.sensordata[10:]    #state_E
        tip_velocity = data.sensordata[:3]
        state_G = get_external_forces(model, data)
        state_R = get_friction_forces(model, data)
        
        # ---------------- Note that Framequat sensor returns 4 values [x,y,z,w]
        rel_quat = data.sensordata[3:7]
        w = rel_quat[3]
        theta = 2 * np.arccos(w)
        theta = np.clip(theta, -np.pi/2, np.pi/2)
        
        mj.mj_forward(model, data)
        mj.mj_step(model, data)
        time +=dt

    

    #print(data.site_xpos[0])
    
    if (seq>=N):
        pass
    # if (data.time>=simend):
    #     break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
os.execl(sys.executable, *sys.argv)