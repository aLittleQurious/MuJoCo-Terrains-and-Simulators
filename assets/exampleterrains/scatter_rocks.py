import xml.etree.ElementTree as ET
import random

src_file = "rocky_terrain_scene.xml"
n = 3000 #number
size_range = [0.05, 0.15] #Size
scatter_range = [-4, 4] #This is a 2d box in which the rocks are scattered
height_range = [0, 0.02] #This is the height at which the rocks are scattered
quat_range = [0, 1] #Rotation angles

"""This works by finding the xml_file, which is assumed to have a 'rocky_terrain_body' tag, and begins appending a bunch of scattered boxes, which act like rocks"""
def append_rocks(src_file, dst_file, n, size_range, scatter_range, height_range, quat_range):
    tree = ET.parse(src_file)
    root = tree.getroot()
    rocky_terrain_body = root.find(".//body[@name='rocky_terrain_body']")
    
    if rocky_terrain_body is None:
        raise ValueError("Invalid MuJoCo XML: Missing rocky_terrain_body element.")
    
    """Randomly generate a rock geom"""
    for i in range(n):
        size = [random.uniform(size_range[0], size_range[1]) for _ in range(3)]
        scatter_position = [random.uniform(scatter_range[0], scatter_range[1]) for _ in range(2)]
        height_position = [random.uniform(height_range[0], height_range[1]) for _ in range(1)]
        
        
        quat = [random.uniform(quat_range[0], quat_range[1]) for _ in range(4)]
        quat = [q / sum(quat) for q in quat]  # Normalize quaternion

        rock = ET.Element("geom", {
            "name": f"rock{i+1}",
            "type": "box",
            "size": f"{size[0]} {size[1]} {size[2]}",
            "pos": f"{scatter_position[0]} {scatter_position[1]} {height_position[0]}",
            "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
            "rgba": "0.5 0.4 0.3 1"
        })
        
        rocky_terrain_body.append(rock)
    
    tree.write(dst_file)
    print(f"Appended {n} rocks to {src_file} and saved as modified_{dst_file}.")
    

    
if __name__ == "__main__":
    dst_file = "a_destination.xml"
    
    append_rocks(src_file, dst_file, n, size_range, scatter_range, height_range, quat_range)