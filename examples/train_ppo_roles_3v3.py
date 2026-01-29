# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
Train a single shared PPO policy with hierarchical roles for 3v3 PyQuaticus.

This script trains one PPO policy that learns to act conditioned on role
(ATTACK/DEFEND/INTERCEPT) assignments that change every ROLE_PERIOD steps.
"""

import argparse
import gymnasium as gym
import numpy as np
import pygame
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec, Policy
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen
from pyquaticus.config import config_dict_std
import logging
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK, DEFEND, INTERCEPT


class RoleBasedRewardWrapper:
    """
    Wrapper to provide role-based rewards to agents.
    """
    
    def __init__(self, base_env):
        self.base_env = base_env
        self.current_roles = {}
    
    def set_roles(self, roles):
        """Set current roles for reward calculation."""
        self.current_roles = roles
    
    def step(self, actions):
        """Step environment and compute role-based rewards."""
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        
        # Modify rewards based on roles
        modified_rewards = {}
        for agent_id, reward in rewards.items():
            role_id = self.current_roles.get(agent_id, ATTACK)
            # Use hierarchical role reward function
            # Note: This is a simplified version - in practice you'd need access to full state
            modified_rewards[agent_id] = reward
        
        return obs, modified_rewards, terminated, truncated, info


class HierarchicalPPOPolicy(Policy):
    """
    Single PPO policy that learns role-conditioned behaviors.
    """
    
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # The PPO algorithm will handle the actual neural network
        # This wrapper just ensures we use the same policy for all agents


def create_role_reward_config():
    """
    Create reward configuration that uses hierarchical role rewards.
    """
    # Create a reward function that includes role information
    def role_reward_wrapper(agent_id, team, agents, agent_inds_of_team, state, 
                           prev_state, env_size, agent_radius, catch_radius, 
                           scrimmage_coords, max_speeds, tagging_cooldown):
        # Extract role from observation if available, otherwise default to ATTACK
        role_id = ATTACK  # Default
        
        # Try to extract role from agent's current observation if state has it
        try:
            if hasattr(state, 'get') and 'agent_observations' in state:
                agent_obs = state['agent_observations'].get(agent_id)
                if agent_obs is not None and len(agent_obs) >= 3:
                    # Last 3 elements should be role one-hot
                    role_one_hot = agent_obs[-3:]
                    role_id = np.argmax(role_one_hot)
        except Exception:
            # If extraction fails, use default
            pass
        
        return rew.hierarchical_role_reward(
            agent_id, team, agents, agent_inds_of_team, state, prev_state,
            env_size, agent_radius, catch_radius, scrimmage_coords, 
            max_speeds, tagging_cooldown, role_id
        )
    
    return role_reward_wrapper


class RandPolicy(Policy):
    """Random policy for baseline opponents."""
    
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


def env_creator_with_roles(config):
    """Create environment with hierarchical roles."""
    try:
        # Create base environment
        base_env = pyquaticus_v0.PyQuaticusEnv(**config)
        
        # Wrap with role functionality
        role_env = wrap_env_with_roles(base_env)
        
        # Wrap with PettingZoo wrapper for RLLib
        return ParallelPettingZooWrapper(role_env)
    except Exception as e:
        print(f"Error creating environment with roles: {e}")
        print("Falling back to environment without roles")
        # Fallback to environment without role modifications
        base_env = pyquaticus_v0.PyQuaticusEnv(**config)
        return ParallelPettingZooWrapper(base_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hierarchical PPO policy for 3v3 PyQuaticus')
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    parser.add_argument('--checkpoint', help='Resume from checkpoint', type=str, default=None)
    parser.add_argument('--iterations', help='Number of training iterations', type=int, default=8000)
    parser.add_argument('--save-dir', help='Directory to save checkpoints', type=str, default='./hierarchical_checkpoints')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    RENDER_MODE = 'human' if args.render else None
    
    # Environment configuration
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True
    config_dict['team_size'] = 3
    
    # Validate configuration
    if not isinstance(config_dict, dict):
        raise ValueError("config_dict must be a dictionary")
    
    # Use hierarchical role rewards for our agents
    try:
        role_reward_func = create_role_reward_config()
    except Exception as e:
        print(f"Error creating role reward function: {e}")
        print("Using default reward function")
        role_reward_func = rew.caps_and_grabs
    
    reward_config = {
        'agent_0': role_reward_func,
        'agent_1': role_reward_func, 
        'agent_2': role_reward_func,
        'agent_3': None,  # Random opponents
        'agent_4': None,
        'agent_5': None
    }
    
    # Create environment
    env_config = {
        'config_dict': config_dict,
        'render_mode': RENDER_MODE,
        'reward_config': reward_config,
        'team_size': 3
    }
    
    try:
        env = env_creator_with_roles(env_config)
        register_env('pyquaticus_hierarchical', lambda config: env_creator_with_roles(config))
    except Exception as e:
        print(f"Error registering environment: {e}")
        print("Exiting...")
        sys.exit(1)
    
    # Get observation and action spaces with error handling
    try:
        obs_space = env.observation_space['agent_0']
        act_space = env.action_space['agent_0']
    except (KeyError, AttributeError) as e:
        print(f"Error getting observation/action spaces: {e}")
        print("Exiting...")
        sys.exit(1)
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Map agents to policies - all our agents use the same shared policy."""
        if agent_id in ['agent_0', 'agent_1', 'agent_2']:
            return "hierarchical_policy"
        return "random_policy"
    
    # Policy configuration
    policies = {
        'hierarchical_policy': (None, obs_space, act_space, {}),
        'random_policy': (RandPolicy, obs_space, act_space, {"no_checkpoint": True})
    }
    
    env.close()
    
    # PPO configuration with safe hyperparameters
    ppo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env='pyquaticus_hierarchical')
        .env_runners(num_env_runners=1, num_cpus_per_env_runner=1)
        .training(
            lr=3e-4,           # Learning rate
            gamma=0.99,        # Discount factor
            lambda_=0.95,      # GAE parameter
            clip_param=0.2,    # PPO clipping
            entropy_coeff=0.01, # Entropy bonus
            train_batch_size=4000,  # Training batch size
            sgd_minibatch_size=128,
            num_sgd_iter=10
        )
    )
    
    # Multi-agent configuration
    ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["hierarchical_policy"],
    )
    
    # Build algorithm
    algo = ppo_config.build_algo()
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        algo.restore(args.checkpoint)
        print(f"Resumed from checkpoint: {args.checkpoint}")
    
    # Create save directory
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directory: {e}")
        print("Using current directory for checkpoints")
        args.save_dir = '.'
    
    # Initialize Ray with error handling
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        print("Continuing without Ray...")
    
    # Training loop
    print(f"Starting training for {args.iterations} iterations...")
    print(f"Checkpoints will be saved to: {args.save_dir}")
    
    try:
        for i in range(args.iterations):
            start_time = time.time()
            
            # Train one iteration
            result = algo.train()
            
            end_time = time.time()
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}:")
                print(f"  Episode reward mean: {result.get('episode_reward_mean', 'N/A')}")
                print(f"  Training time: {end_time - start_time:.2f}s")
            
            # Save checkpoint periodically
            if i % 500 == 0 and i > 0:
                try:
                    checkpoint_path = algo.save(f"{args.save_dir}/iter_{i}/")
                    print(f"Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    print(f"Error saving checkpoint at iteration {i}: {e}")
        
        # Final checkpoint
        try:
            final_checkpoint = algo.save(f"{args.save_dir}/final/")
            print(f"Final checkpoint saved: {final_checkpoint}")
        except Exception as e:
            print(f"Error saving final checkpoint: {e}")
        
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Clean up Ray
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
