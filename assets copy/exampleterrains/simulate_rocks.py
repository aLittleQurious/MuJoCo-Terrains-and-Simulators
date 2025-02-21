import mujoco
from mujoco import MjSim, MjViewer

model = mujoco.MjModel.from_xml_path("rocky_terrain_scene.xml")
sim = MjSim(model)
viewer = MjViewer(sim)

while True:
    sim.step()
    viewer.render()
