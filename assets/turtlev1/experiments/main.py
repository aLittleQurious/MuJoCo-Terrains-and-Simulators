import os
import yaml

path_to_yaml = "" #Provided by the user from the command line

# Load the configuration
with open(path_to_yaml, "r") as file:
    config = yaml.safe_load(file)

# Extract file names
robot_src_file = config["robot_model"]
terrain_file = config["terrain_model"]

# Construct full file paths
robot_src_file_path = os.path.join(os.getcwd(), config["paths"]["template_dir"], robot_src_file)
terrain_file_path = os.path.join(os.getcwd(), config["paths"]["saved_dir"], terrain_file)



# Print values to verify
print(f"Source File Path: {robot_src_file_path}")
print(f"Terrain File Path: {terrain_file_path}")
print(f"Parameters: {params}")

