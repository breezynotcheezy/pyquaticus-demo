import numpy as np
import sys
import os
import os.path
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team

# Configuration for the environment - only use recognized parameters
config_dict = {
    "max_time": 600.0,
    "sim_speedup_factor": 8,
    "max_score": 100,
    "render_agent_ids": True
}

# Create the environment
env = PyQuaticusEnv(
    team_size=1,  # 1v1
    config_dict=config_dict,
    render_mode='human'  # Set to 'rgb_array' for headless
)

# Reset the environment
obs, info = env.reset()

# Get agent IDs from the environment
agent_ids = env.possible_agents

try:
    step = 0
    while True:
        
        # Simple policy: move forward with slight turns
        # [forward_velocity, angular_velocity]
        blue_action = [1.0, 0.5]  # Move forward and turn right
        red_action = [0.7, -0.3]  # Move forward and turn left
        
        # Create action dictionary
        actions = {}
        for agent_id in agent_ids:
            # For continuous actions, use [forward_velocity, angular_velocity]
            if 'blue' in agent_id:
                actions[agent_id] = [1.0, 0.5]  # Move forward and turn right
            else:
                actions[agent_id] = [4, -0.3]  # Move forward and turn left
        
        # Take a step
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
        if step > 1000:  # Prevent infinite loops
            print("Max steps reached")
            break
            
except KeyboardInterrupt:
    print("Test stopped by user")
    
finally:
    # Always close the environment
    env.close()