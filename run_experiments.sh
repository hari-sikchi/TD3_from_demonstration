#!/bin/bash

# Script to reproduce results
python main_multistep.py --policy "TD3" --env "Hopper-v2" --seed 100 --start_timesteps 1000 --exp_name "action-repeat-only-explore-training" &


python main.py --policy "TD3" --env "HalfCheetah-v2" --seed 100 --start_timesteps 10000 --exp_name "baseline-MDP-model" --save_model & 

python main.py --policy "TD3" --env "Swimmer-v2" --seed 100 --start_timesteps 1000 --exp_name "BC_initialization_10_episodes" --load_model "Hopper-v2_clone-model_100" & 


python main_clone.py --policy "TD3" --env "Swimmer-v2" --seed 100 --start_timesteps 1000 --exp_name "clone-model-1-ep" --save_model & 


python main_prior.py --policy "TD3" --env "Swimmer-v2" --seed 500 --start_timesteps 1000 --exp_name "prior_load_BC_10_episodes_initialize" --load_model "Swimmer-v2_clone-model_100" & 

python main_act_curr.py --policy "TD3" --env "HalfCheetah-v2" --seed 500 --start_timesteps 10000 --exp_name "bayesian-UCB" &


python main_episodic_multistep.py --policy "TD3" --env "Hopper-v2" --seed 500 --start_timesteps 1000 --exp_name "lookahead_5_lambda_weighted_corrected" &

for ((i=0;i<5;i+=1))
do 
	# python main.py \
	# --policy "TD3" \
	# --env "HalfCheetah-v2" \
	# --seed $i \
	# --start_timesteps 10000

	python main.py \
	--policy "TD3" \
	--env "Hopper-v2" \
	--seed $i \
	--start_timesteps 1000 \
	--exp_name "persistent" &

	# python main.py \
	# --policy "TD3" \
	# --env "Walker2d-v2" \
	# --seed $i \
	# --start_timesteps 1000

	# python main.py \
	# --policy "TD3" \
	# --env "Ant-v2" \
	# --seed $i \
	# --start_timesteps 10000

	# python main.py \
	# --policy "TD3" \
	# --env "InvertedPendulum-v2" \
	# --seed $i \
	# --start_timesteps 1000

	# python main.py \
	# --policy "TD3" \
	# --env "InvertedDoublePendulum-v2" \
	# --seed $i \
	# --start_timesteps 1000

	# python main.py \
	# --policy "TD3" \
	# --env "Reacher-v2" \
	# --seed $i \
	# --start_timesteps 1000
done
