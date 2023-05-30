import torch
import os
from panda_pushing_env import PandaImageSpacePushingEnv
from dataloader_and_controller import *
from utils import *
from SINDyModel import *


print("Estimated Running time: 5 - 10 mins")
print("NOTE: The controller some times samples a trajectory that does not achieve Goal. In this case, it will restart Simulation")
data = torch.load('model_weights.pt')
model = SINDyModel(action_dim=3, order=2, latent_dim=16, num_channels=1, trig_functions=True)
model.load_state_dict(data)

collected_data = np.load('collected_data_100.npy', allow_pickle=True)
train_loader, val_loader, norm_constants = process_data_multiple_step(collected_data, batch_size=500, num_steps=1)
norm_tr = NormalizationTransform(norm_constants)

target_state = np.array([0.7, 0., 0.])
goal_reached = False

while not goal_reached:
    print("Attempting to reach")
    env = PandaImageSpacePushingEnv(visualizer=None, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True)
    state_0 = env.reset()
    env.object_target_pose = env._planar_pose_to_world_pose(target_state)
    controller = PushingLatentController(env, model, latent_space_pushing_cost_function,norm_constants, num_samples=100, horizon=10)

    state = state_0

    # num_steps_max = 100
    num_steps_max = 20

    for i in range(num_steps_max):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        # check if we have reached the goal
        end_pose = env.get_object_pos_planar()
        goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
        goal_reached = goal_distance < BOX_SIZE
        if done or goal_reached:
            print("Steps: ",i)
            break

    print(f'GOAL REACHED: {goal_reached}')
    if not goal_reached:
        print("Resetting environment")
        
# plt.close(fig)

