import numpy as np
import torch


class EpisodicReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e3),max_episodes = int(1e3)):
		self.max_size = max_size
		self.max_episodes = max_episodes
		self.episode_ptr = 0
		self.ptr = 0
		self.size = np.zeros(max_episodes)
		self.episode_size = 0
		self.state = np.zeros((max_episodes,max_size, state_dim))
		self.action = np.zeros((max_episodes,max_size, action_dim))
		self.next_state = np.zeros((max_episodes,max_size, state_dim))
		self.reward = np.zeros((max_episodes,max_size, 1))
		self.not_done = np.zeros((max_episodes,max_size, 1))
  
  
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.episode_ptr,self.ptr] = state
		self.action[self.episode_ptr,self.ptr] = action
		self.next_state[self.episode_ptr,self.ptr] = next_state
		self.reward[self.episode_ptr,self.ptr] = reward
		self.not_done[self.episode_ptr,self.ptr] = 1. - done
  
		if done:
			self.ptr = 0
			self.episode_ptr = ([self.episode_ptr]+1)% self.max_episodes
			self.episode_size = min(self.episode_size + 1, self.max_episodes)
		self.ptr = (self.ptr + 1) % self.max_size
		self.size[self.episode_ptr] = min(self.size[self.episode_ptr] + 1, self.max_size)
		


	def sample(self, batch_size,lookaheads = 5,discount_factor = 0.99):
		episode_ind = np.random.randint(0, self.episode_size+1, size=batch_size)
		row_ind = []
		for row in episode_ind:
			row_ind.append(np.random.randint(0, self.size[row]))
			# indices.append([row,np.random.randint(0, self.size[row])])
   
		# ind = np.random.randint(0, self.size, size=batch_size)
		row_ind = np.array(row_ind)
		row_ind_orig = row_ind.copy()

  
		rewards = torch.FloatTensor(self.reward[episode_ind,row_ind]).to(self.device)*0
		results = []
		for i in range(lookaheads):
			# print(row_ind)
			rewards+=pow(discount_factor,i)*torch.FloatTensor(self.reward[episode_ind,row_ind]).to(self.device)
			results.append(
       		[torch.FloatTensor(self.state[episode_ind,row_ind_orig]).to(self.device),
			torch.FloatTensor(self.action[episode_ind,row_ind_orig]).to(self.device),
			torch.FloatTensor(self.next_state[episode_ind,row_ind]).to(self.device),
			rewards,
			torch.FloatTensor(self.not_done[episode_ind,row_ind]).to(self.device)])
			for j,ep_ind in enumerate(episode_ind):
				if(row_ind[j]>=self.size[ep_ind]-1):
					continue
				row_ind[j]+=1

		return results
   
   
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
  
		self.next_reward = np.zeros((max_size, 1))
		self.next_action = np.zeros((max_size, action_dim))
		self.next_next_state = np.zeros((max_size, state_dim))
		self.next_not_done = np.zeros((max_size, 1))

  
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done, next_reward = -100.0,next_action=0.0, next_next_state=0.0,next_done = False):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.next_next_state[self.ptr] = next_next_state
		self.next_reward[self.ptr] = next_reward
		self.next_action[self.ptr] = next_action
		self.next_not_done[self.ptr] = 1. - next_done
  
  

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size,multistep = False):
		ind = np.random.randint(0, self.size, size=batch_size)
		if(multistep):
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.FloatTensor(self.action[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.not_done[ind]).to(self.device),
				torch.FloatTensor(self.next_reward[ind]).to(self.device),
				torch.FloatTensor(self.next_action[ind]).to(self.device),
				torch.FloatTensor(self.next_next_state[ind]).to(self.device),
				torch.FloatTensor(self.next_not_done[ind]).to(self.device),
    
			)			
		else:
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.FloatTensor(self.action[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.not_done[ind]).to(self.device)
			)