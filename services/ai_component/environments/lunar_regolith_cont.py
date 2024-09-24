import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
from xml.etree import ElementTree as ET
import tempfile

class LunarRegolithSimulationContinuous(gym.Env):
    """
    Custom MuJoCo Gymnasium environment simulating a lunar regolith arena with specified zones.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 robot_sdl_path,
                 arena_length=10.0,
                 arena_width=10.0,
                 material_friction=1.0,
                 material_density=1.0,
                 num_motors=12,
                 num_craters=5,
                 num_boulders=10,
                 goal_area_size=(1.0, 1.0, 0.5)):
        """
        Initialize the simulation environment.

        Parameters:
        - robot_sdl_path (str): Path to the robot's SDL file.
        - arena_length (float): Length of the arena in the x-direction.
        - arena_width (float): Width of the arena in the y-direction.
        - material_friction (float): Friction coefficient simulating lunar regolith.
        - material_density (float): Density of the lunar regolith material.
        - num_craters (int): Number of craters in the navigation zone.
        - num_boulders (int): Number of boulders in the navigation zone.
        - action_type (str): 'continuous' or 'discrete' action space.
        - goal_area_size (tuple): Dimensions of the goal area (length, width, height).
        """
        super(LunarRegolithSimulationContinuous, self).__init__()

        # Store parameters
        self.arena_length = arena_length
        self.arena_width = arena_width
        self.material_friction = material_friction
        self.material_density = material_density
        self.num_craters = num_craters
        self.num_boulders = num_boulders
        self.goal_area_size = goal_area_size  # (length, width, height)

        # Load robot model
        self.robot_sdl_path = robot_sdl_path
        if not os.path.exists(self.robot_sdl_path):
            raise FileNotFoundError(f"Robot SDL file not found at {self.robot_sdl_path}")

        # Create MuJoCo XML
        self.model_xml = self.create_mujoco_xml()

        # Initialize MuJoCo simulation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp:
            tmp.write(self.model_xml)
            tmp_path = tmp.name

        self.model = mujoco.load_model_from_path(tmp_path)
        self.sim = mujoco.MjSim(self.model)
        self.viewer = None

        # Define action and observation spaces
        if self.action_type == 'continuous':
            # Example: continuous actions for joint torques
            self.num_motors = num_motors
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_motors,), dtype=np.float32)
        elif self.action_type == 'discrete':
            # Example: discrete actions (e.g., forward, backward, left, right, stop)
            self.action_space = spaces.Discrete(5)
        else:
            raise ValueError("action_type must be 'continuous' or 'discrete'")

        # Example observation space: robot's position and velocity
        obs_low = np.full(self.sim.model.nq + self.sim.model.nv, -np.inf)
        obs_high = np.full(self.sim.model.nq + self.sim.model.nv, np.inf)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Define Goal Area boundaries for reward calculation
        # Positioned within the Construction Zone
        construction_center_x = self.arena_length / 4  # Right half
        construction_center_y = 0  # Centered along y-axis
        construction_center_z = self.goal_area_size[2] / 2  # Assuming ground level z=0

        self.goal_area_bounds = {
            'x_min': construction_center_x - self.goal_area_size[0] / 2,
            'x_max': construction_center_x + self.goal_area_size[0] / 2,
            'y_min': construction_center_y - self.goal_area_size[1] / 2,
            'y_max': construction_center_y + self.goal_area_size[1] / 2,
            'z_min': 0,
            'z_max': self.goal_area_size[2]
        }

        # Track boulders that have been counted towards the reward
        self.counted_boulders = set()

        # Remove the temporary XML file
        os.remove(tmp_path)

    def create_mujoco_xml(self):
        """
        Create the MuJoCo XML string based on parameters.
        """
        # Parse the robot SDL file
        robot_tree = ET.parse(self.robot_sdl_path)
        robot_root = robot_tree.getroot()

        # Create the main XML structure
        mujoco = ET.Element('mujoco', model='lunar_regolith_arena')

        # Compiler settings
        compiler = ET.SubElement(mujoco, 'compiler', angle='degree', coordinate='local')

        # Optionally add visualization settings
        option = ET.SubElement(mujoco, 'option', timestep='0.001', gravity='0 0 -9.81')

        # Define default material simulating lunar regolith
        default_material = ET.SubElement(mujoco, 'asset')
        material = ET.SubElement(default_material, 'material', name='lunar_regolith',
                                 friction=str(self.material_friction),
                                 density=str(self.material_density))

        # Define the arena (ground plane)
        worldbody = ET.SubElement(mujoco, 'worldbody')

        # Ground plane with lunar regolith material
        geom_ground = ET.SubElement(worldbody, 'geom', name='ground',
                                    type='plane',
                                    size=f"{self.arena_length/2} {self.arena_width/2} 0.1",
                                    material='lunar_regolith',
                                    pos='0 0 0')

        # Add massive central column
        column = ET.SubElement(worldbody, 'geom', name='central_column',
                               type='cylinder',
                               size='0.5 2.0',  # radius, height
                               pos='0 0 1.0',  # assuming height starts from ground
                               material='lunar_regolith')

        # Define Navigation Zone (Left Half)
        nav_zone = ET.SubElement(worldbody, 'site', name='navigation_zone',
                                 pos=f"{-self.arena_length/4} 0 0",
                                 size=f"{self.arena_length/2} {self.arena_width} 0",
                                 rgba='0.8 0.8 0.8 0')

        # Add craters
        for i in range(self.num_craters):
            crater = ET.SubElement(worldbody, 'geom', name=f'crater_{i}',
                                   type='sphere',
                                   size='0.5',
                                   pos=f"{np.random.uniform(-self.arena_length/2 + 1.0, 0 - 1.0)} "
                                       f"{np.random.uniform(-self.arena_width/2 + 1.0, self.arena_width/2 - 1.0)} "
                                       f"0.25",
                                   material='lunar_regolith',
                                   contype='1', conaffinity='1',
                                   rgba='0.5 0.5 0.5 1')

        # Add boulders
        for i in range(self.num_boulders):
            boulder = ET.SubElement(worldbody, 'geom', name=f'boulder_{i}',
                                    type='sphere',
                                    size='0.3',
                                    pos=f"{np.random.uniform(-self.arena_length/2 + 1.0, 0 - 1.0)} "
                                        f"{np.random.uniform(-self.arena_width/2 + 1.0, self.arena_width/2 - 1.0)} "
                                        f"0.3",
                                    material='lunar_regolith',
                                    contype='1', conaffinity='1',
                                    rgba='0.6 0.6 0.6 1')

        # Define Excavation Zone (Right Upper Half)
        excavation_zone = ET.SubElement(worldbody, 'site', name='excavation_zone',
                                        pos=f"{self.arena_length/4} {self.arena_width/4} 0",
                                        size=f"{self.arena_length/2} {self.arena_width/2} 0",
                                        rgba='0.2 0.8 0.2 0.3')  # Semi-transparent

        # Define Construction Zone (Right Lower Half)
        construction_zone = ET.SubElement(worldbody, 'site', name='construction_zone',
                                          pos=f"{self.arena_length/4} {-self.arena_width/4} 0",
                                          size=f"{self.arena_length/2} {self.arena_width/2} 0",
                                          rgba='0.2 0.2 0.8 0.3')  # Semi-transparent

        # Define Goal Area within Construction Zone
        goal_area = ET.SubElement(worldbody, 'geom', name='goal_area',
                                   type='box',
                                   size=f"{self.goal_area_size[0]/2} {self.goal_area_size[1]/2} {self.goal_area_size[2]/2}",
                                   pos=f"{self.arena_length/4} {-self.arena_width/4} {self.goal_area_size[2]/2}",
                                   rgba='1 0 0 0.5',  # Semi-transparent red
                                   material='lunar_regolith',
                                   contype='0', conaffinity='0')  # Non-collidable

        # Insert the robot model into the worldbody
        worldbody.append(robot_root)

        # Close the XML
        xml_string = ET.tostring(mujoco, encoding='unicode')
        return xml_string

    def step(self, action):
        """
        Execute one simulation step.

        Parameters:
        - action: Action to be taken by the agent.

        Returns:
        - observation: Agent's observation after the action.
        - reward: Reward obtained.
        - done: Whether the episode has ended.
        - info: Additional information.
        """
        self.sim.data.ctrl[:] = action
        # Step the simulation
        self.sim.step()

        # Get observation
        observation = self.get_observation()

        # Compute reward
        reward = self.compute_reward()

        # Check if done
        done = self.check_done()

        # Additional info
        info = {}

        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        """
        Reset the simulation to an initial state.

        Returns:
        - observation: Initial observation.
        - info: Additional information.
        """
        super().reset(seed=seed)
        self.sim.reset()
        observation = self.get_observation()
        info = {}
        # Reset the counted boulders
        self.counted_boulders = set()
        return observation, info

    def render(self, mode='human'):
        """
        Render the simulation.

        Parameters:
        - mode (str): Rendering mode.

        Returns:
        - If mode is 'rgb_array', returns the image as an array.
        """
        if self.viewer is None:
            self.viewer = mujoco.MjViewer(self.sim)

        if mode == 'human':
            self.viewer.render()
        elif mode == 'rgb_array':
            # Obtain the RGB array from the viewer
            width, height, depth = self.sim.render(640, 480, camera_name=None)
            return self.sim.render(width, height, camera_name=None)
        else:
            raise NotImplementedError("Render mode not supported.")

    def close(self):
        """
        Clean up the environment.
        """
        if self.viewer is not None:
            self.viewer = None

    def get_observation(self):
        """
        Retrieve the current observation from the simulation.

        Returns:
        - observation (np.array): Current state observation.
        """
        # Example: concatenate position and velocity
        qpos = self.sim.data.qpos.flatten()
        qvel = self.sim.data.qvel.flatten()
        observation = np.concatenate([qpos, qvel])
        return observation

    def compute_reward(self):
        """
        Compute the reward for the current state.

        Returns:
        - reward (float): Computed reward based on volume in the goal area.
        """
        reward = 0.0

        # Iterate through all boulders
        for i in range(self.num_boulders):
            boulder_name = f"boulder_{i}"
            try:
                boulder_id = self.sim.model.geom_name2id(boulder_name)
            except ValueError:
                # Boulder not found
                continue

            # Get boulder's position
            boulder_pos = self.sim.data.geom_xpos[boulder_id]

            # Check if boulder is within the Goal Area bounds
            if (self.goal_area_bounds['x_min'] <= boulder_pos[0] <= self.goal_area_bounds['x_max'] and
                self.goal_area_bounds['y_min'] <= boulder_pos[1] <= self.goal_area_bounds['y_max'] and
                self.goal_area_bounds['z_min'] <= boulder_pos[2] <= self.goal_area_bounds['z_max']):
                
                if i not in self.counted_boulders:
                    # Calculate volume of the boulder (assuming sphere)
                    radius = self.sim.model.geom_size[boulder_id][0]
                    volume = (4/3) * np.pi * (radius ** 3)
                    reward += volume

                    # Mark this boulder as counted
                    self.counted_boulders.add(i)

        return reward

    def check_done(self):
        """
        Determine if the episode is done.

        Returns:
        - done (bool): Whether the episode has ended.
        """
        # Placeholder termination condition
        # TODO: Define based on task objectives, e.g., time limit or full accumulation
        # Example: Episode ends when all boulders are in the goal area
        done = len(self.counted_boulders) == self.num_boulders
        return done