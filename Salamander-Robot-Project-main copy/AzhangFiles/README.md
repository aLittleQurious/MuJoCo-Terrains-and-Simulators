
# Salamander Reinforcement Learning

This example uses a simple custom MuJoCo model and trains a reinforcement learning agent using the Gymnasium shell and algorithms from StableBaselines. The Ball Balance environment should be changed to the Salamander environment later.

---

## How to Create and Use a Virtual Environment in Python

Virtual environments allow you to create isolated Python environments for your projects, helping you manage dependencies and avoid conflicts between them.

### Step 1: Install `venv` (if not already installed)

The `venv` module is included by default in Python 3.3 and later. To ensure it's available:
1. Verify Python is installed by running:
   ```bash
   python --version
   ```
2. Install Python if it’s not already installed:
   - [Download Python](https://www.python.org/downloads/).

### Step 2: Create a Virtual Environment

1. Open your terminal or command prompt.
2. Navigate to your project directory:
   ```bash
   cd /path/to/your/project
   ```
3. Create the virtual environment by running:
   ```bash
   python -m venv venv_name
   ```
   Replace `venv_name` with your desired name (commonly just `venv`).

### Step 3: Activate the Virtual Environment

- **On Windows:**
   ```bash
   venv_name\Scripts\activate
   ```
- **On macOS/Linux:**
   ```bash
   source venv_name/bin/activate
   ```

Once activated, you’ll see the virtual environment name in your terminal prompt, indicating it’s active.

### Step 4: Install Packages

While in the virtual environment, you can install packages using `pip`:
```bash
pip install package_name
```

To reproduce the results, you will need the Python packages [MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html), [Gymnasium](https://gymnasium.farama.org/index.html), and [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) with the appropriate versions.

You can use the `requirements.txt` provided to install all the dependencies:
```bash
pip install -r requirements.txt
```

### Step 5: Deactivate the Virtual Environment

When you're done, deactivate the virtual environment by running:
```bash
deactivate
```

---

## How to Use the Current Code?

### Step 1: MuJoCo Files

1. The model in MJCF format is located in the [assets](https://github.com/MiniroLab/Salamander-Robot-Project/tree/main/AzhangFiles/assets) folder. Currently, the XML for the Ball Balance environment is being used. However, the Salamander XML and other needed assets are placed in the folder and can be easily replaced in the [Environment] code.

2. By running the [model_viewer.py](https://github.com/MiniroLab/Salamander-Robot-Project/blob/main/AzhangFiles/model_viewer.py) file, you can test all actuators, as well as the physics of the model in general, in a convenient interactive mode. To do this, run the following command:
   ```bash
   python model_viewer.py --path path/towards/your/xml/file
   ```

### Step 2: Training an Agent

1. In the file [learn.py](https://github.com/MiniroLab/Salamander-Robot-Project/blob/main/AzhangFiles/learn.py), you can change parameters related to learning or change the algorithm used for the RL agent. Make sure to enter the name of the algorithm and the model number so the code logs the results and models correctly. You can train multiple models at the same time based on the device you are using. After changing the model name and number in the code, open a new terminal and run the command:
   ```bash
   python learn.py
   ```

### Step 3: Logging the Training Process

TensorBoard is a tool for visualizing machine learning training metrics, such as loss, accuracy, and more, in real-time.

1. Run TensorBoard with the following command:
   ```bash
   tensorboard --logdir=logs
   ```
2. Once TensorBoard starts, it will provide a URL (e.g., `http://localhost:6006`). Open this URL in your web browser to view the training metrics.

### Step 4: Testing the Agent

1. In the file [test.py](https://github.com/MiniroLab/Salamander-Robot-Project/blob/main/AzhangFiles/test.py), you can test your agent by specifying the path to the model saved after training. Models are saved in the folder [models](). You can also create a GIF from frames (commented code).

2. Create your own environment class similar to [BallBalanceEnv](https://github.com/MiniroLab/Salamander-Robot-Project/blob/main/AzhangFiles/ball_balance_env.py).

   - In the `__init__` method, replace the model path with your own, and insert your observation shape into `observation_space` (size of observation).
   - In the `step` method, define the reward for each step, as well as the condition for the end of the episode.
   - In the `reset_model` method, define what should happen at the beginning of each episode (when the model is reset). For example, initial states, adding noise, etc.
   - In the `_get_obs` method, return your observations, such as the velocities and coordinates of certain joints.

3. In the file [learn.py](https://github.com/MiniroLab/Salamander-Robot-Project/blob/main/AzhangFiles/learn.py), create an instance of your class (instead of BallBalanceEnv). Then, you can choose a different algorithm or use your own; now your environment has all the qualities of the Gym environment.

4. In the file [test.py](https://github.com/MiniroLab/Salamander-Robot-Project/blob/main/AzhangFiles/test.py), you can test your agent by specifying the path to the model saved after training. You can also create a GIF from frames (commented code).

