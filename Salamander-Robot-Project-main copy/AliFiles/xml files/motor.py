import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys

xml_path = 'azhang.xml' #xml file (assumes this is in the same folder as this file)
simend = 10 #simulation time
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

def gait_pass(sequence):
    global gaits
    
    spinal_joint_angle= gaits[0][sequence]
    leg1_joint_angle=gaits[1][sequence]
    leg2_joint_angle=gaits[2][sequence]
    leg3_joint_angle=gaits[3][sequence]
    leg4_joint_angle=gaits[4][sequence]
    shoulder1_joint_angle=gaits[5][sequence]
    shoulder2_joint_angle=gaits[6][sequence]
    shoulder3_joint_angle=gaits[7][sequence]
    shoulder4_joint_angle=gaits[8][sequence]
    
    return [spinal_joint_angle,
            leg1_joint_angle,
            leg2_joint_angle,
            leg3_joint_angle,
            leg4_joint_angle,
            shoulder1_joint_angle,
            shoulder2_joint_angle,
            shoulder3_joint_angle,
            shoulder4_joint_angle
            ]

def get_joint_id(model, joint_name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    global seq
    gaits = gait_pass(seq)
    data.ctrl[get_joint_id(model, "spinal_joint")-1] = 0
    #data.ctrl[get_joint_id(model, "spinal_joint")-1] = gaits[0]
    data.ctrl[get_joint_id(model, "leg1_joint")-1] = gaits[1]
    data.ctrl[get_joint_id(model, "leg2_joint")-1] = gaits[2]
    data.ctrl[get_joint_id(model, "leg3_joint")-1] = gaits[3]
    data.ctrl[get_joint_id(model, "leg4_joint")-1] = gaits[4]
    data.ctrl[get_joint_id(model, "shoulder1_joint")-1] = gaits[5]
    data.ctrl[get_joint_id(model, "shoulder2_joint")-1] = gaits[6]
    data.ctrl[get_joint_id(model, "shoulder3_joint")-1] = gaits[7]
    data.ctrl[get_joint_id(model, "shoulder4_joint")-1] = gaits[8]
    
    if seq == 247:
        seq = 0
    else:
        seq += 1
    

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

cam.azimuth = 90
cam.elevation = -45
cam.distance = 3
cam.lookat = np.array([1.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)
spinal = np.array([])
i = True
ball_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ball")
site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "tip_sensors")
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/300.0):
        site_pos = data.site_xpos[site_id]
        ball_pos = data.geom_xpos[ball_id]
        distance = np.linalg.norm(ball_pos[:2] - site_pos[:2])
        print(distance)
        mj.mj_step(model, data)

    if (data.time>=simend):
        break
        np.save('data.npy', spinal)

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
