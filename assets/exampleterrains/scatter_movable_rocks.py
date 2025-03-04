import xml.etree.ElementTree as ET
import random

src_file = "template_terrain_movable.xml" #source file
dst_directory = "rocky_terrains" #Destination directory of the numbere of rocky terrain variations
dst_filename_prefix = "movable_terrain" #Within the directory, this decides the filename
num_copies = 10 #Number of rocky terrain variations you'd like to generate with parameters below

n = 500 #number
size_range = [0.005, 0.015] #Size
scatter_range = [-0.25, 0.25] #This is a 2d box in which the rocks are scattered
height_range = [0.05, 0.1] #This is the height at which the rocks are scattered
mass_range = [0.00001, 0.0001]
quat_range = [0, 1] #Rotation angles
type = "ellipsoid" #Type of Rock



"""This works by finding the xml_file, which is assumed to have a 'rocky_terrain_body' tag, and begins appending a bunch of scattered boxes, which act like rocks."""
def append_rocks(src_file, dst_file, n, size_range, scatter_range, height_range, quat_range, type="box", mass_range=[0.00001, 0.0001]):
    tree = ET.parse(src_file)
    root = tree.getroot()
    rocky_terrain_body = root.find(".//worldbody") #We have to append to worldbody for free joint
    
    if rocky_terrain_body is None:
        raise ValueError("Invalid MuJoCo XML: Missing rocky_terrain_body element.")
    
    """Randomly generate a rock body with a free joint"""
    for i in range(n):
        size = [random.uniform(size_range[0], size_range[1]) for _ in range(3)]
        scatter_position = [random.uniform(scatter_range[0], scatter_range[1]) for _ in range(2)]
        height_position = [random.uniform(height_range[0], height_range[1]) for _ in range(1)]
        mass_range = [random.uniform(mass_range[0], height_range[1]) for _ in range(1)]
        
        quat = [random.uniform(quat_range[0], quat_range[1]) for _ in range(4)]
        quat = [q / sum(quat) for q in quat]  # Normalize quaternion

        rock_body = ET.Element("body", {"name": f"rock{i+1}", "pos": f"{scatter_position[0]} {scatter_position[1]} {height_position[0]}"})
        
        rock_geom = ET.Element("geom", {
            "type": type,
            "size": f"{size[0]} {size[1]} {size[2]}",
            "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
            "rgba": "0.5 0.4 0.3 1",

        })
        
        joint = ET.Element("joint", {
            "type": "free"
        })
        
        rock_body.append(joint)
        rock_body.append(rock_geom)
        rocky_terrain_body.append(rock_body)
    
    tree.write(dst_file)
    print(f"Appended {n} rocks to {src_file} and saved as {dst_file}.")
    
    
if __name__ == "__main__":
    
    for i in range(num_copies): #create n copies of this terrain
    
        append_rocks(src_file, f"{dst_filename_prefix}{i}.xml", n, size_range, scatter_range, height_range, quat_range, type)
