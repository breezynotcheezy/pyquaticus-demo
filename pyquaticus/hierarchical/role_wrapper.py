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
Environment wrapper for hierarchical role-based observations.

This wrapper intercepts observations and appends role one-hot encoding
to enable a single PPO policy to learn role-conditioned behaviors.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from gymnasium.spaces import Box
from pyquaticus.structs import Team
from pyquaticus.hierarchical.roles import (
    ROLE_PERIOD, choose_role, get_team_roles, 
    role_to_one_hot, append_roles_to_observations
)

class HierarchicalRoleWrapper:
    """
    Wrapper that adds role-based conditioning to observations.
    
    This wrapper modifies observations to include role one-hot encoding,
    allowing a single policy to learn different behaviors conditioned on role.
    """
    
    def __init__(self, env):
        """
        Initialize the role wrapper.
        
        Args:
            env: The base PyQuaticus environment
        """
        self.env = env
        self.step_count = 0
        self.current_roles = {}
        self.last_role_update = 0
        
        # Validate environment has required attributes
        if not hasattr(env, 'possible_agents'):
            raise AttributeError("Environment must have 'possible_agents' attribute")
        if not hasattr(env, 'observation_space'):
            raise AttributeError("Environment must have 'observation_space' method")
        
        # Store original observation spaces
        self.original_observation_spaces = {}
        for agent_id in env.possible_agents:
            try:
                self.original_observation_spaces[agent_id] = env.observation_space(agent_id)
            except Exception as e:
                print(f"Warning: Could not get observation space for {agent_id}: {e}")
                continue
        
        # Modified observation spaces with role one-hot (3 additional dimensions)
        self.observation_spaces = {}
        for agent_id, orig_space in self.original_observation_spaces.items():
            try:
                orig_shape = orig_space.shape
                new_shape = (orig_shape[0] + 3,) if len(orig_shape) == 1 else (orig_shape[0] + 3,)
                
                # Create new observation space with role dimensions
                new_low = np.concatenate([orig_space.low, np.zeros(3)])
                new_high = np.concatenate([orig_space.high, np.ones(3)])
                self.observation_spaces[agent_id] = Box(
                    low=new_low, 
                    high=new_high, 
                    shape=new_shape,
                    dtype=orig_space.dtype
                )
            except Exception as e:
                print(f"Warning: Could not create modified observation space for {agent_id}: {e}")
                # Use original space as fallback
                self.observation_spaces[agent_id] = orig_space
    
    def observation_space(self, agent_id: str):
        """Return the modified observation space with role encoding."""
        return self.observation_spaces.get(agent_id, self.env.observation_space(agent_id))
    
    def _update_roles(self):
        """Update roles for all agents if enough steps have passed."""
        if self.step_count - self.last_role_update >= ROLE_PERIOD:
            try:
                # Get current state for role assignment
                team_obs = {'state': self.env.state}
                
                # Update roles for each team
                for team in [Team.BLUE_TEAM, Team.RED_TEAM]:
                    if hasattr(self.env, 'agents_of_team') and team in self.env.agents_of_team:
                        team_agents = self.env.agents_of_team[team]
                        agent_ids = [agent.id for agent in team_agents]
                        
                        if agent_ids:  # Only if team has agents
                            team_roles = get_team_roles(team_obs, agent_ids, team)
                            self.current_roles.update(team_roles)
                
                self.last_role_update = self.step_count
            except Exception as e:
                print(f"Warning: Error updating roles: {e}")
                # Continue with existing roles if update fails
    
    def _modify_observations(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Append role one-hot encoding to observations.
        
        Args:
            observations: Original observations from environment
            
        Returns:
            Modified observations with role encoding
        """
        return append_roles_to_observations(observations, self.current_roles)
    
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], 
                                                   Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Step the environment and modify observations to include roles.
        """
        try:
            self.step_count += 1
            
            # Update roles if needed
            self._update_roles()
            
            # Step the environment
            observations, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Modify observations to include roles
            modified_observations = self._modify_observations(observations)
            
            return modified_observations, rewards, terminated, truncated, info
        except Exception as e:
            print(f"Error in step: {e}")
            # Return original observations if modification fails
            return observations, rewards, terminated, truncated, info
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment and initialize roles.
        """
        try:
            self.step_count = 0
            self.last_role_update = 0
            self.current_roles = {}
            
            # Reset the environment
            observations, info = self.env.reset(seed=seed, options=options)
            
            # Initialize roles
            self._update_roles()
            
            # Modify observations to include roles
            modified_observations = self._modify_observations(observations)
            
            return modified_observations, info
        except Exception as e:
            print(f"Error in reset: {e}")
            # Return original observations if modification fails
            return observations, info
    
    def __getattr__(self, name):
        """Delegate any other attribute access to the wrapped environment."""
        return getattr(self.env, name)


def wrap_env_with_roles(env):
    """
    Convenience function to wrap an environment with hierarchical roles.
    
    Args:
        env: PyQuaticus environment to wrap
        
    Returns:
        Environment with role-based observations
    """
    try:
        return HierarchicalRoleWrapper(env)
    except Exception as e:
        print(f"Error creating role wrapper: {e}")
        print("Returning original environment without role modifications")
        return env
