### Robot, Terrain, and Simulator Implementation in MuJoCo

## About 

This is a structured way to implement Robots in MuJoCo with terrains and AI models (Primarily CPGs). It uses Yaml files to promote modularity and simplification of commands.

Furthermore, custom Terrain Generation is included within the `terrain_generation` directory

## Getting Started 

Simply copy the `assets` folder. Once copied, you can either generate your own terrain, or you can run a simulation of your robot on a custom terrain.

## Generating a custom terrain

This makes use of the `terrain_generation` directory. To get started, run main.py paired with a `.yaml` file from the accompanying `yamls` directory.

## Usage:

1. `python main.py`
Or
2. `python main.py basic_terrain.yaml`
If no yaml is specified (like in 1.), the yaml file that will be used will be at the top of `main.py`

The terrain `yamls` have the following structure:

```
# This is the NAME of the template file we build upon. Must be in 
# template terrains
template_file: "template_terrain_fixed.xml"  

# This is the name of our output terrain. It gets placed in the saved_terrains directory
output_file: "my_fixed_terrain.xml"

#Type of terrain. Options: fixed | movable | sloped
type_of_terrain: fixed 

#True/False. this is an extra parameter that allows us to instad save in the experiment directories's terrain models.
#TLDR: You don't have to move the terrain you generated to terrain_models everytime. Options: True | False
save_in_terrain_models: True 

#These are the paramters we set. If not specified, it will use the default values
parameters:
  n: 2000  # Number of rocks to make. Can be any non negative integer.
  size_range: [0.005, 0.015]  # Size. Can be any range
  mass_range: [0.1,  0.5] #Controls the mass of the rocks. Can be any range.
  scatter_range: [-0.5, 0.5]  # Grid area over which the rocks are scattered. Can be any range
  height_range: [0, 0.015]  # Heights at which the rocks are placed. Can be any range.
  quat_range: [0, 1]  # Rotation angles of the rocks. Can only be between [0, 1]
  rock_type: "box"  # Type of rock shape. "box" | "ellipsoid"
  sliding_friction_range: [0.7, 0.8] #Elements in MuJoCo have 3 types of frictions, sliding, torsional, rolling. You can change them here. All can be any range
  torsional_friction_range: [0.9, 1]
  rolling_friction_range: [0.8, 0.9]
```

Running a command like `python main.py basic_terrain.yaml` will generate a terrain. These terrains are created by randomly sampling values from the ranges above and creating a rock from those values. Repeat that `n` times to generate the terrain.

- `/template_terrains` are xml's we build our terrain upon. You don't need to touch them.
- `/saved_terrains` is where our generated terrains are saved.
- `/terrain_generators` contain the core of how terrains are generated. There are different types depending on the terrain.
- `/setup` contains basic setup functions.
- `/yamls` contains the yamls we use to create the terrains.

### Running a Robot in an environment in /experiments

## Overview

Given a robot's xml. (see `example_robot.xml` in `/robot_models), we can run it with a terrain and simulator.

## Prerequisites: Setting up the Robot

Once you have the robot's xml, place the xml inside the `/robot_models` directory and the robot's STL's inside `/model_assets/{name_of_your_robot}`. Note that if you `<include>` other xml's (to promote modularity), place them inside `/robot_models` as well. 

## Prerequisites: Setting up the Terrain

Once you have a terrain you'd like to use, place it inside the `/terrain_models` directory. Then you are done

## Prerequisites: Simulating your robot with a CPG

All AI simulators in MuJoCo rely on a model and data variable. We supply those two variables with the followng code: 

```
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from setup import load_model_from_yaml

if "yaml_path" not in globals(): #if yaml not provided, use model and data from this
    raise ValueError("yaml_path must be provided.")
yaml_path = globals().get("yaml_path")

model, data = load_model_from_yaml.get_model_and_data_from_yaml(yaml_path)
``` 

Hence, to connect your simulator, include this code where you normally define the `model` and `data` variables in your code.

## Usage:

1. `python main.py`
Or
2. `python main.py basic_terrain.yaml`
If no yaml is specified (like in 1.), the yaml file that will be used will be at the top of `main.py`


