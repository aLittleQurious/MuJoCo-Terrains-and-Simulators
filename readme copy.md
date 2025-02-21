.\.venv\Scripts\activate  
To activate venv

python -m mujoco.viewer 
launches an empty visualization session, where a model can be loaded by drag-and-drop.

python -m mujoco.viewer --mjcf=/path/to/some/mjcf.xml 
launches a visualization session for the specified model file.