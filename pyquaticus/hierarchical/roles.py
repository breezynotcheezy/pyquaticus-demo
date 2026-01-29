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
Hierarchical role assignment for PyQuaticus agents.

This module implements rule-based role assignment for multi-agent teams,
enabling hierarchical behavior without complex multi-policy architectures.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from pyquaticus.structs import Team

# Role constants
ROLE_PERIOD = 15  # Steps between role updates

ATTACK = 0
DEFEND = 1  
INTERCEPT = 2

ROLE_NAMES = {
    ATTACK: "ATTACK",
    DEFEND: "DEFEND", 
    INTERCEPT: "INTERCEPT"
}

def role_to_one_hot(role_id: int) -> np.ndarray:
    """Convert role ID to one-hot encoding."""
    if role_id not in [ATTACK, DEFEND, INTERCEPT]:
        role_id = ATTACK  # Default to ATTACK if invalid
    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[role_id] = 1.0
    return one_hot

def choose_role(agent_obs: np.ndarray, team_obs: Dict[str, Any], agent_id: str, 
               team: Team, all_agents: List[str]) -> int:
    """
    Choose role for an agent based on simple rule-based logic.
    
    Args:
        agent_obs: Observation for the specific agent (not used in current implementation)
        team_obs: Team-level observations and state
        agent_id: ID of the agent to assign role to
        team: Team enum (BLUE_TEAM or RED_TEAM)
        all_agents: List of all agent IDs in the team
        
    Returns:
        Role ID (ATTACK, DEFEND, or INTERCEPT)
    """
    # Validate inputs
    if not all_agents or agent_id not in all_agents:
        return ATTACK
    
    if not isinstance(team_obs, dict) or 'state' not in team_obs:
        return ATTACK
    # Extract relevant state information with error handling
    state = team_obs.get('state', {})
    
    # Initialize default values
    team_has_flag = False
    agent_has_flag = []
    flag_positions = []
    agent_positions = []
    
    # Safely extract state information
    try:
        if 'team_has_flag' in state and len(state['team_has_flag']) > team.value:
            team_has_flag = state['team_has_flag'][team.value]
        
        if 'agent_has_flag' in state:
            agent_has_flag = state['agent_has_flag']
        
        if 'flag_position' in state and len(state['flag_position']) >= 2:
            flag_positions = state['flag_position']
        
        if 'agent_position' in state:
            agent_positions = state['agent_position']
    except (IndexError, TypeError, KeyError):
        # If there's any error accessing state, default to ATTACK
        return ATTACK
    
    # Get agent indices with bounds checking
    try:
        agent_idx = all_agents.index(agent_id)
        teammate_indices = [all_agents.index(aid) for aid in all_agents]
    except ValueError:
        return ATTACK  # Agent ID not found in list
    
    # Determine flag positions with bounds checking
    try:
        enemy_flag_idx = 1 - team.value  # Opposite team's flag
        home_flag_idx = team.value
        
        if len(flag_positions) <= max(enemy_flag_idx, home_flag_idx):
            return ATTACK
            
        enemy_flag_pos = flag_positions[enemy_flag_idx]
        home_flag_pos = flag_positions[home_flag_idx]
    except (IndexError, TypeError):
        return ATTACK
    
    # Current agent position with bounds checking
    try:
        if len(agent_positions) <= agent_idx:
            return ATTACK
        agent_pos = agent_positions[agent_idx]
    except (IndexError, TypeError):
        return ATTACK
    
    # Validate positions are not None
    if agent_pos is None or enemy_flag_pos is None or home_flag_pos is None:
        return ATTACK
    
    # Calculate distances
    def distance(pos1, pos2):
        """Calculate distance between two positions safely."""
        try:
            if pos1 is None or pos2 is None:
                return float('inf')
            return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
        except (TypeError, ValueError):
            return float('inf')
    
    # Case 1: Team has enemy flag
    if team_has_flag:
        # Find flag carrier
        flag_carrier_idx = None
        for i, teammate_idx in enumerate(teammate_indices):
            if len(agent_has_flag) > teammate_idx and agent_has_flag[teammate_idx]:
                flag_carrier_idx = teammate_idx
                break
        
        if flag_carrier_idx is not None:
            # Flag carrier: ATTACK (return behavior)
            if agent_idx == flag_carrier_idx:
                return ATTACK
            
            # Closest teammate to carrier: INTERCEPT (escort)
            carrier_pos = agent_positions[flag_carrier_idx]
            dist_to_carrier = distance(agent_pos, carrier_pos)
            
            # Check if this agent is closest to carrier (excluding carrier)
            is_closest = True
            for other_idx in teammate_indices:
                if other_idx != flag_carrier_idx and other_idx != agent_idx:
                    other_pos = agent_positions[other_idx]
                    if distance(other_pos, carrier_pos) < dist_to_carrier:
                        is_closest = False
                        break
            
            if is_closest:
                return INTERCEPT
            
            # Remaining teammate: DEFEND (protect home)
            return DEFEND
    
    # Case 2: No one has flag
    else:
        # Calculate distances for all teammates with error handling
        try:
            dists_to_enemy_flag = []
            dists_to_home_flag = []
            
            for teammate_idx in teammate_indices:
                if len(agent_positions) <= teammate_idx:
                    # If teammate position is unavailable, use large distance
                    dists_to_enemy_flag.append(float('inf'))
                    dists_to_home_flag.append(float('inf'))
                    continue
                    
                teammate_pos = agent_positions[teammate_idx]
                dists_to_enemy_flag.append(distance(teammate_pos, enemy_flag_pos))
                dists_to_home_flag.append(distance(teammate_pos, home_flag_pos))
            
            if not dists_to_enemy_flag or not dists_to_home_flag:
                return ATTACK
            
            # Closest to enemy flag: ATTACK
            min_enemy_dist_idx = np.argmin(dists_to_enemy_flag)
            if agent_idx == min_enemy_dist_idx:
                return ATTACK
            
            # Closest to home flag: DEFEND  
            min_home_dist_idx = np.argmin(dists_to_home_flag)
            if agent_idx == min_home_dist_idx:
                return DEFEND
            
            # Remaining: INTERCEPT
            return INTERCEPT
            
        except (ValueError, IndexError):
            return ATTACK

def get_team_roles(team_obs: Dict[str, Any], agent_ids: List[str], team: Team) -> Dict[str, int]:
    """
    Get roles for all agents in a team.
    
    Args:
        team_obs: Team-level observations
        agent_ids: List of agent IDs in the team
        team: Team enum
        
    Returns:
        Dictionary mapping agent_id -> role_id
    """
    roles = {}
    
    # Get observations for each agent (simplified - using team_obs for all)
    for agent_id in agent_ids:
        # In a real implementation, you'd pass the actual agent observation
        agent_obs = np.array([])  # Placeholder
        role = choose_role(agent_obs, team_obs, agent_id, team, agent_ids)
        roles[agent_id] = role
    
    return roles

def append_roles_to_observations(observations: Dict[str, np.ndarray], 
                                 roles: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Append role one-hot encoding to each agent's observation.
    
    Args:
        observations: Dictionary of agent_id -> observation
        roles: Dictionary of agent_id -> role_id
        
    Returns:
        Updated observations with role one-hot appended
    """
    updated_obs = {}
    
    for agent_id, obs in observations.items():
        try:
            role_id = roles.get(agent_id, ATTACK)  # Default to ATTACK if not found
            role_one_hot = role_to_one_hot(role_id)
            
            # Handle different observation formats
            if obs is None:
                # Create minimal observation if None
                obs_array = np.zeros(10, dtype=np.float32)
            else:
                obs_array = np.asarray(obs, dtype=np.float32)
            
            # Append role one-hot to observation
            if len(obs_array.shape) == 1:
                updated_obs[agent_id] = np.concatenate([obs_array, role_one_hot])
            else:
                # Handle multi-dimensional observations (flatten first, then append)
                flat_obs = obs_array.flatten()
                updated_obs[agent_id] = np.concatenate([flat_obs, role_one_hot])
                
        except Exception as e:
            # If any error occurs, create a default observation with role
            print(f"Warning: Error processing observation for {agent_id}: {e}")
            role_one_hot = role_to_one_hot(ATTACK)
            default_obs = np.concatenate([np.zeros(10, dtype=np.float32), role_one_hot])
            updated_obs[agent_id] = default_obs
    
    return updated_obs
