import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Conv2d
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from mppi import MPPI
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE
# import control

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random_trajectory(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, 32, 32, num_channels) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = []
    # --- Your code here
    for i in range(num_trajectories):
      state = env.reset()
      states = np.zeros((trajectory_length+1,) + env.observation_space.shape, dtype=np.uint8)
      actions = np.zeros((trajectory_length,) + env.action_space.shape, dtype=np.float32)
      

      for j in range(trajectory_length):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        states[j] = state
        actions[j] = action

        state = next_state

        if done:
          break

      states[-1] = state
      
      collected_data.append({'states': states, 'actions': actions})



    # ---
    return collected_data


class NormalizationTransform(object):

    def __init__(self, norm_constants):
        self.norm_constants = norm_constants
        self.mean = norm_constants['mean']
        self.std = norm_constants['std']

    def __call__(self, sample):
        """
        Transform the sample by normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        # --- Your code here
        #### APPROACH 1 ####
        sample['states'] = self.normalize_state(sample['states'])

        #### APPROACH 2 ####
        # states = sample['states']
        # norm_states = (states - self.mean)/self.std
        # sample['states'] = norm_states

        # ---
        return sample

    def inverse(self, sample):
        """
        Transform the sample by de-normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        # --- Your code here
        sample['states'] = self.denormalize_state(sample['states'])
        # ---
        return sample

    def normalize_state(self, state):
        """
        Normalize the state using the provided normalization constants.
        :param state: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        # --- Your code here
        state = (state - self.mean)/self.std
        # ---
        return state

    def denormalize_state(self, state_norm):
        """
        Denormalize the state using the provided normalization constants.
        :param state_norm: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        # --- Your code here
        state = (state_norm*self.std) + self.mean
        # ---
        return state


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'states': x_t,x_{t+1}, ... , x_{t+num_steps}
     'actions': u_t, ..., u_{t+num_steps-1},
    }
    where:
     states: torch.float32 tensor of shape (batch_size, num_steps+1, state_size)
     actions: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.

    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.

    :return train_loader: <torch.utils.data.DataLoader> for training
    :return val_loader: <torch.utils.data.DataLoader> for validation
    :return normalization_constants: <dict> containing the mean and std of the states.

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_data = None
    val_data = None
    normalization_constants = {
        'mean': None,
        'std': None,
    }
    # Your implemetation needs to do the following:
    #  1. Initialize dataset
    #  2. Split dataset,
    #  3. Estimate normalization constants for the train dataset states.
    # --- Your code here
    dataset = MultiStepDynamicsDataset(collected_data, num_steps)
    train_data,val_data = random_split(dataset, [0.8,0.2])
    
    states = train_data[0]['states']
    
    for i in range(1,len(train_data)):
      states = torch.cat((states,train_data[i]['states']),dim=0)

    mean = torch.mean(states)
    std = torch.std(states)
    
    normalization_constants['mean'] = mean
    normalization_constants['std'] = std
    # ---
    norm_tr = NormalizationTransform(normalization_constants)
    train_data.dataset.transform = norm_tr
    val_data.dataset.transform = norm_tr

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)


    return train_loader, val_loader, normalization_constants


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.
    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'states':[x_{t}, x_{t+1},..., x_{t+num_steps} ] -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'actions': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
    }

    Observation: If num_steps=1, this dataset is equivalent to SingleStepDynamicsDataset.
    """

    def __init__(self, collected_data, num_steps=4, transform=None):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps
        self.transform = transform

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (states, actions).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'states': None,
            'actions': None,
        }
        # --- Your code here       
        traj_index = item // self.trajectory_length
        state_index = item % self.trajectory_length
        
        sample['states'] = torch.from_numpy(self.data[traj_index]["states"][state_index:state_index + self.num_steps +1])
        sample['actions'] = torch.from_numpy(self.data[traj_index]["actions"][state_index:state_index + self.num_steps])
        
        if self.transform is not None:
            sample = self.transform(sample) 

        sample['states'] = sample['states'].to(torch.float32)
        sample['actions'] = sample['actions'].to(torch.float32)
        sample['states'] = sample['states'].permute(0,3,1,2)
        # ---
        return sample


def img_space_pushing_cost_function(state, action, target_state):
    """
    Compute the state cost for MPPI on a setup without obstacles in state space (images).
    :param state: torch tensor of shape (B, w, h, num_channels)
    :param action: torch tensor of shape (B, action_size)
    :param target_state: torch tensor of shape (w, h, num_channels)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    cost = None
    # --- Your code here
    B = state.shape[0]
    
    # loss = nn.MSELoss(reduction='sum')
    target_state = target_state.unsqueeze(0)
    # print(state.shape)
    # print(target_state.shape)
    
    target_state = target_state.repeat(state.shape[0],1,1,1)
    # print(state.shape)
    # print(target_state.shape)
    #target_state = target_state.unsqueeze(0).repeat(B, 1, 1, 1)  
    # mse_loss = torch.mean((state - target_state)**2, dim=(1,2,3))
    # cost = mse_loss
    cost = torch.mean((state-target_state)**2,dim=(1,2,3))
    # print(cost.shape)

    # ---
    return cost

class PushingImgSpaceController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, norm_constants, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.norm_constants = norm_constants
        self.target_state = torch.as_tensor(self.env.get_target_state(), dtype=torch.float32).permute(2, 0, 1)
        self.target_state_norm = (self.target_state - self.norm_constants['mean']) / self.norm_constants['std']
        self.cost_function = cost_function
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.1 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, wrapped_state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, wrapped_state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        unwrap_state = self._unwrap_state(state)
        next_state = self.model(unwrap_state, action)
        next_state = self._wrap_state(next_state)
        # ---
        return next_state

    def _compute_costs(self, state, action):
        """
        Compute the cost for each state-action pair.
        You need to call self.cost_function to compute the cost.
        :param state: torch tensor of shape (B, wrapped_state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
        """
        cost = None
        # --- Your code here
        state = self._unwrap_state(state)
        cost = self.cost_function(state, action, self.target_state_norm)


        # ---
        return cost

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (height, width, num_channels) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be sent to the mppi controller. Note that MPPI works with torch tensors.
         - You may need to normalize the state to the same space used for training the model.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state).float()#.unsqueeze(0)
        #print(state_tensor.shape)
        #state_tensor = self.norm_constants.normalize(state_tensor)
        #print(self.norm_constants)
        mean = self.norm_constants['mean']
        std = self.norm_constants['std']
        state_tensor = (state_tensor - mean) / std
        state_tensor = state_tensor.permute(2,0,1)
        wrapped_state = self._wrap_state(state_tensor)

        # ---
        action_tensor = self.mppi.command(wrapped_state)
        # --- Your code here
        action = action_tensor.detach().numpy()



        # ---
        return action

    def _wrap_state(self, state):
        # convert state from shape (..., num_channels, height, width) to shape (..., num_channels*height*width)
        wrapped_state = None
        # --- Your code here
        batch_shape = state.shape[:-3]
        num_channels = state.shape[-3]
        height = state.shape[-2]
        width = state.shape[-1]
        wrapped_state = state.reshape(*batch_shape, num_channels*height*width)



        # ---
        return wrapped_state

    def _unwrap_state(self, wrapped_state):
        # convert state from shape (..., num_channels*height*width) to shape (..., num_channels, height, width)
        state = None
        # --- Your code here
        batch_shape = wrapped_state.shape[:-1]
        num_channels = 1 #self.env.observation_space.shape[0][-3]
        height = 32#self.env.observation_space.shape[0][-2]
        width = 32#self.env.observation_space.shape[0][-1]
        state = wrapped_state.reshape(*batch_shape, num_channels, height, width)


        # ---
        return state

def latent_space_pushing_cost_function(latent_state, action, target_latent_state):
    """
    Compute the state cost for MPPI on a setup without obstacles in latent space.
    :param state: torch tensor of shape (B, latent_dim)
    :param action: torch tensor of shape (B, action_size)
    :param target_latent_state: torch tensor of shape (latent_dim,)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    cost = None
    # --- Your code here
    # model = LatentDynamicsModel(latent_dim=latent_state.shape[1], action_dim=action.shape[1], num_channels=3)
    # next_latent_state = model.latent_dynamics(latent_state, action)
    # print(next_latent_state.shape)
    cost_func = nn.MSELoss()
    # print(latent_state.shape)
    # print(target_latent_state.repeat(latent_state.shape[0],1).shape)

    #target_latent_state = target_latent_state.repeat(latent_state.shape[0],1)

    #cost = torch.mean((target_latent_state - latent_state)**2,dim=1)

    target_latent_state = target_latent_state.repeat(latent_state.shape[0],1)
    cost = (latent_state - target_latent_state) @ (latent_state - target_latent_state).T
    cost = torch.diagonal(cost,0)

    # cost = torch.diagonal(cost,0)
    # cost = torch.linalg.norm(latent_state - target_latent_state.repeat(latent_state.shape[0],1))
    # print(cost.shape)


    # ---
    return cost

class PushingLatentController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, norm_constants, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.norm_constants = norm_constants
        self.target_state = torch.as_tensor(self.env.get_target_state(), dtype=torch.float32).permute(2, 0, 1)
        self.target_state_norm = (self.target_state - self.norm_constants['mean']) / self.norm_constants['std']
        self.latent_target_state = self.model.encode(self.target_state_norm)
        self.cost_function = cost_function
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = 16#model.latent_dim  # Note that the state size is the latent dimension of the model
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.15 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.05
        # ---
        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, latent_dim)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, latent_dim) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        #print(state.shape)
        mask = None
        curr_state = torch.cat((state,action),dim=1)
        library = self.model.SINDy(curr_state)
        next_state = self.model.SINDy.evaluate(library)
        



        # ---
        return next_state

    def _compute_costs(self, state, action):
        """
        Compute the cost for each state-action pair.
        You need to call self.cost_function to compute the cost.
        :param state: torch tensor of shape (B, latent_dim)
        :param action: torch tensor of shape (B, action_size)
        :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
        """
        cost = None
        # --- Your code here
        cost = self.cost_function(state, action, self.latent_target_state)



        # ---
        return cost

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (height, width, num_channels) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be sent to the mppi controller. Note that MPPI works with torch tensors.
         - You may need to normalize the state to the same space used for training the model.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state).float()#.unsqueeze(0)
        #print(state_tensor.shape)
        #state_tensor = self.norm_constants.normalize(state_tensor)
        #print(self.norm_constants)
        mean = self.norm_constants['mean']
        std = self.norm_constants['std']
        state_tensor = (state_tensor - mean) / std
        state_tensor = state_tensor.permute(2,0,1)
        latent_state = self.model.encode(state_tensor)
        #print(latent_state.shape)


        # ---
        action_tensor = self.mppi.command(latent_state)
        # --- Your code here
        action = action_tensor.detach().numpy()
        
        #action = np.array([action_tensor[0][i] for i in self.action_indices])



        # ---
        return action

