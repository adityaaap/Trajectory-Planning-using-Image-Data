import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# from numpngw import write_apng
# from IPython.display import Image
# from tqdm.notebook import tqdm


class Encoder(nn.Module):
    def __init__(self,latent_dim=16,num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        
        self.conv_layers = nn.Sequential(
          nn.Conv2d(self.num_channels, 4, kernel_size=5, stride=1, padding=0),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=0),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Flatten(),
          nn.Linear(100,100),
          nn.ReLU(inplace=True),
          nn.Linear(100, self.latent_dim),
       )
        
    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return: 2 <torch.Tensor>
          :mu: <torch.Tensor> of shape (..., latent_dim)
          :log_var: <torch.Tensor> of shape (..., latent_dim)
        """
        latent_state = None
        input_shape = state.shape
        state = state.reshape(-1, self.num_channels, 32, 32)
        latent_state = self.conv_layers(state)
        latent_state = latent_state.reshape(*input_shape[:-3], self.latent_dim)
        return latent_state


class Decoder(nn.Module):

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # --- Your code here
        self.layers = nn.Sequential(
          nn.Linear(self.latent_dim, 500),
          nn.ReLU(inplace=True),
          nn.Linear(500,500),
          nn.ReLU(inplace=True),
          nn.Linear(500, self.num_channels*(32**2)),
        )


    def forward(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return decoded_state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        """
        decoded_state = None
        input_shape = latent_state.shape
        # print("Latent state NaN?: ",torch.isnan(latent_state).any())
        latent_state = latent_state.reshape(-1, self.latent_dim)
        decoded_state = self.layers(latent_state)
        # print("Decoded state NaN?: ",torch.isnan(decoded_state).any())
        decoded_state = decoded_state.reshape(*input_shape[:-1], self.num_channels, 32, 32)
        # print("Final Decoded state NaN?: ",torch.isnan(decoded_state).any())
        return decoded_state


class SINDyLibrary(nn.Module):
    def __init__(self,latent_dim,action_dim, order, trig_functions = True):
        super().__init__()
        self.n_dim = latent_dim+action_dim
        self.order = order
        self.trig_fn = trig_functions
        self.library = None
        self.epsilon = nn.Parameter(torch.ones((self._countDim(), latent_dim)))
        # self.epsilon = nn.Parameter(torch.rand((self._countDim(), latent_dim)))

    def forward(self,z):
        """
        Args:
        z: Latent space of dim (B,latent_dim+action_dim)
        
        Output:
        library: library of functions (B, X)  
        """
        library = [torch.ones((z.shape[0],1))]

        for i in range(self.n_dim):
             library.append(z[:,i].reshape(-1,1))
        
        if (self.order>1):
            for i in range(self.n_dim):
                for j in range(i,self.n_dim):
                    library.append(torch.mul(z[:,i], z[:,j]).reshape(-1,1))
        if (self.order>2):
            for i in range(self.n_dim):
                for j in range(i,self.n_dim):
                    for k in range(j,self.n_dim):
                        library.append((z[:,i]*z[:,j]*z[:,k]).reshape(-1,1))

        if self.trig_fn:
            for i in range(self.n_dim):
                library.append(torch.sin(z[:,i]).reshape(-1,1))
        # print("Library shape: ",torch.stack(library).shape)
        # library = torch.transpose(torch.stack(library),dim0=0,dim1=1)
        library = torch.cat(library,dim=1)
        # print("Library shape: ",library.shape)
       
        return library

    #### ABSOULTE ####
    def evaluate(self, library, coeff_mask=None):
        if coeff_mask is not None:
            
            return torch.matmul(library,coeff_mask*self.epsilon)
        else:
            return torch.matmul(library, self.epsilon)

    #### RESIDUAL ####
    # def evaluate(self, library, state, coeff_mask=None):
    #     if coeff_mask is not None:
    #         return state + torch.matmul(library, coeff_mask*self.epsilon)
    #     else:
    #         return state + torch.matmul(library, self.epsilon)


    def _countDim(self):
        count = 1 + self.n_dim
        if (self.order>1):
            for i in range(self.n_dim):
                for j in range(i,self.n_dim):
                    count+=1
        if (self.order>2):
            for i in range(self.n_dim):
                for j in range(i,self.n_dim):
                    for k in range(j,self.n_dim):
                        count+=1

        if self.trig_fn:
            for i in range(self.n_dim):
                count+=1
        return count


class SINDyModel(nn.Module):
    def __init__(self, action_dim, order, latent_dim=16, num_channels=1, trig_functions = True):
        super().__init__()
        self.encode = Encoder(latent_dim,num_channels)
        self.decode = Decoder(latent_dim, num_channels)
        self.SINDy = SINDyLibrary(latent_dim,action_dim, order, trig_functions)
    def forward(self,states,actions,coeff_mask=None):
        #print(states.shape)
        encoded_state = self.encode(states)
        curr_state = torch.cat((encoded_state,actions),dim=1)
        library = self.SINDy(curr_state)
        ## Residual
        # next_latent_state = self.SINDy.evaluate(library, encoded_state, coeff_mask)
        ## Absoulte
        next_latent_state = self.SINDy.evaluate(library,coeff_mask)
        next_state = self.decode(next_latent_state)
        return next_state

class SINDyLoss(nn.Module):
    def __init__(self, state_loss_fn, latent_loss_fn, alpha1=0.1, alpha2=0.1, alpha3=0.1):
        super().__init__()
        self.state_loss = state_loss_fn
        self.latent_loss = latent_loss_fn
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def forward(self, model, states, actions, coeff_mask = None):
      rec_loss = 0.
      encoded_states = model.encode(states)
    #   print("Encoded states: ",encoded_states.shape)
      decoded_states = model.decode(encoded_states)
      rec_loss = self.state_loss(decoded_states, states)
    #   print("REC LOSS: ", rec_loss)


      pred_latent_values = []
      pred_states = []
     
      prev_z = encoded_states[:, 0, :]
      
      prev_state = states[:, 0, :]  # get initial state value
      
      for t in range(actions.shape[1]):
          next_z = None
          next_state = None
          
          curr_state = torch.cat((prev_z,actions[:,t,:]), dim=1)
          
          SINDy_library = model.SINDy(curr_state)
          
          if coeff_mask is not None:
              ## Absolute
              next_z = model.SINDy.evaluate(SINDy_library, coeff_mask)
              next_state = model(prev_state,actions[:,t,:],coeff_mask)
              ## Residual
            #   next_z = model.SINDy.evaluate(SINDy_library,prev_z, coeff_mask)
            #   next_state = model.decode(model.SINDy.evaluate(SINDy_library, True, coeff_mask))
              
          else:
              ## Absolute
              next_z = model.SINDy.evaluate(SINDy_library)
              next_state = model(prev_state,actions[:,t,:])
              ## Residual
            #   next_z = model.SINDy.evaluate(SINDy_library,prev_z)
            #   next_state = model.decode(model.SINDy.evaluate(SINDy_library))
              
          
        #   print("Next state NaNs?: ",torch.isnan(next_state).any())
          pred_latent_values.append(next_z)
          pred_states.append(next_state)
      
        
          prev_z = next_z
          prev_state = next_state
      pred_states = torch.stack(pred_states, dim=1)
    #   print("pred_states NaNs?: ",torch.isnan(pred_states).any())
      pred_latent_values = torch.stack(pred_latent_values, dim=1)
      
      # compute prediction loss -- compares predicted state values with the given states
      pred_loss = 0.
      pred_loss = self.state_loss(pred_states, states[:,1:,:,:,:])


      # compute latent loss -- compares predicted latent values with the encoded latent values for states
      lat_loss = 0.


      #lat_loss += self.latent_loss(pred_latent_values, model.encode(states[:, 1:, :, :, :]))
      lat_loss = self.latent_loss(pred_latent_values, encoded_states[:,1:,:])

      # ---

      multi_step_loss = rec_loss + self.alpha1*pred_loss + self.alpha2*lat_loss + self.alpha3*torch.linalg.norm(model.SINDy.epsilon)

      return multi_step_loss
