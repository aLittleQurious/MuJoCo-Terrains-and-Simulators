import os
import setup.load_model_from_yaml as load_model_from_yaml

#Input a yaml file to consume
yaml_file = "turtle_and_fixed.yaml"



robot_model_path, terrain_model_path = load_model_from_yaml.find_model_files(yaml_file)
simulation_path = load_model_from_yaml.find_simulation_file(yaml_file)


print(robot_model_path, terrain_model_path)
print(simulation_path)