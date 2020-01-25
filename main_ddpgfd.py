import numpy as np
import torch
import gym
import argparse
import os
import copy
import utils
import TD3_ddpgfd as TD3
import OurDDPG
import DDPG
import time

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over {} episodes: {}".format(eval_episodes,avg_reward))
	print("---------------------------------------")
	return avg_reward

def save_demonstration(policy, env_name, seed,demonstration_buffer, demonstration_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(demonstration_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			next_state, reward, done, _ = eval_env.step(action)
			demonstration_buffer.add(state, action, next_state, reward, float(done))		
			state = next_state
			avg_reward+= reward

	avg_reward /= demonstration_episodes

	print("---------------------------------------")
	print("Demonstrations| Evaluation over {} episodes: {}".format(demonstration_episodes,avg_reward))
	print("---------------------------------------")
	return demonstration_buffer


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--exp_name", default="dump")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--oracle", default="dump")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--oracle_episodes", default=100)                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--lambda_coeff", default=10)                 # Model load file name, "" doesn't load, "default" uses file_name

	args = parser.parse_args()

	file_name = "{}_{}_{}".format(args.env,args.exp_name,args.seed)
	log_dir = file_name+ "/"
	os.makedirs("./results/"+log_dir, exist_ok=True)
	print("---------------------------------------")
	print("Policy: {}, Env: {}, Seed: {}".format(args.policy,args.env,args.seed))
	print("Experiment name: {}".format(file_name))
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
		oracle_policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load("./models/{}".format(policy_file))
		policy.actor_target=copy.deepcopy(policy.actor)

	oracle_policy.load("./models/{}".format(args.oracle))
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	demonstration_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	oracle_buffer = save_demonstration(oracle_policy, args.env, args.seed,demonstration_buffer,demonstration_episodes=int(args.oracle_episodes))

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	log_data = {'evaluations': [] , 'timesteps': [0]}
	evaluations = [eval_policy(policy, args.env, args.seed)]

	print("Evaluation of initial policy is: {}".format(evaluations[0]))
	time.sleep(1)
	# exit()
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	persistent_MDP = False
	persistent_MDP_curriculum = False
	total_timesteps = 0
	# log_data['timesteps'].append(total_timesteps)
	probs = [0.5,0.25,0.25]
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1
		total_timesteps+=1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer

		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size,env=env,oracle_buffer=oracle_buffer)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps+1,episode_num+1,episode_timesteps,episode_reward))
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			log_data['evaluations']=evaluations
			log_data['timesteps'].append(total_timesteps)
			np.save("./results/"+log_dir+"log_data.npy",log_data)
			# np.save("./results/{}".format(file_name), evaluations)
			if args.save_model: policy.save("./models/{}".format(file_name))
