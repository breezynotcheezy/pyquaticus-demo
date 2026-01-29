import sys
import os
import numpy as np
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team

# Configuration for the environment - only use recognized parameters
config_dict = {
    "max_time": 600.0,
    "max_score": 100,
    "sim_speedup_factor": 8,
    "render_agent_ids": True
}

# Create the environment
env = PyQuaticusEnv(
    team_size=2,  # 2v2
    config_dict=config_dict,
    render_mode='human'  # Set to 'rgb_array' for headless
)

# Reset the environment
obs, info = env.reset()

# Get agent IDs
agent_ids = env.possible_agents
blue_agents = [agent_id for agent_id in agent_ids if 'blue' in agent_id]
red_agents = [agent_id for agent_id in agent_ids if 'red' in agent_id]
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender

# Initialize policies for each agent
r_one = BaseAttacker(red_agents[0], mode='competition_easy')
r_two = BaseDefender(red_agents[1], mode='competition_easy')
b_one = BaseDefender(blue_agents[0], mode='competition_easy')
b_two = BaseAttacker(blue_agents[1], mode='competition_easy')

# Set up the main loop
step = 0
max_steps = 1000
try:
    while step < max_steps:
        # Compute actions for all agents
        actions = {}
        
        # Red team actions
        if red_agents:  # Check if there are red agents
            actions[red_agents[0]] = r_one.compute_action(obs)
            if len(red_agents) > 1:
                actions[red_agents[1]] = r_two.compute_action(obs)
        
        # Blue team actions
        if blue_agents:  # Check if there are blue agents
            actions[blue_agents[0]] = b_one.compute_action(obs)
            if len(blue_agents) > 1:
                actions[blue_agents[1]] = b_two.compute_action(obs)
        
        # Take a step in the environment
        step_return = env.step(actions)
        
        # Handle different return formats
        if len(step_return) == 5:  # New format: obs, reward, done, truncated, info
            next_obs, rewards, dones, truncated, infos = step_return
        elif len(step_return) == 4:  # Old format: obs, reward, done, info
            next_obs, rewards, dones, infos = step_return
            truncated = {agent_id: False for agent_id in agent_ids}
        
        # Update observations
        obs = next_obs
        
        # Check if any agent is done
        if any(dones.values()):
            print("Episode finished!")
            break
            
        step += 1
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{max_steps}")
            
except KeyboardInterrupt:
    print("Test stopped by user")
    
finally:
    # Always close the environment
    env.close()
    print("Environment closed.")
