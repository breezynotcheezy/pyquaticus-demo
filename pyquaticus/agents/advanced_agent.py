import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import random
from collections import deque, namedtuple
import os

from ..base_policies.base_policy import BaseAgentPolicy, EnvType
from pyquaticus.envs.pyquaticus import Team, PyQuaticusEnv

# Hyperparameters (reduced to prevent crashes)
GAMMA = 0.99
LR = 0.0003
TAU = 1e-3
BUFFER_SIZE = int(1e5)  # Reduced from 1M to 100K
BATCH_SIZE = 256       # Reduced from 1024 to 256
UPDATE_EVERY = 4
LEARN_NUM = 1

# Reduce network size
FC1_UNITS = 256  # Reduced from 512
FC2_UNITS = 256  # Reduced from 512

# Prioritized Experience Replay
ALPHA = 0.6
BETA = 0.4
BETA_INC_PER_SAMPLING = 0.001
PRIO_EPSILON = 1e-6

# Noisy Nets
NOISY_STD = 0.1

# Distributional RL
N_ATOMS = 51
V_MIN = -10
V_MAX = 10

# Multi-step Learning
N_STEPS = 3

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""
    def __init__(self, in_features, out_features, std_init=NOISY_STD):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class DuelingNetwork(nn.Module):
    """Dueling Network Architecture"""
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):  # Reduced network size
        super(DuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.LayerNorm(fc1_units),
            nn.SiLU()
        )
        
        # Dueling streams
        self.advantage = nn.Sequential(
            NoisyLinear(fc1_units, fc2_units),
            nn.LayerNorm(fc2_units),
            nn.SiLU(),
            NoisyLinear(fc2_units, action_size)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(fc1_units, fc2_units),
            nn.LayerNorm(fc2_units),
            nn.SiLU(),
            NoisyLinear(fc2_units, 1)
        )
    
    def forward(self, state):
        x = self.feature(state)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        for layer in [self.advantage, self.value]:
            if hasattr(layer[0], 'reset_noise'):
                layer[0].reset_noise()
            if hasattr(layer[3], 'reset_noise'):
                layer[3].reset_noise()

class MultiStepBuffer:
    """N-step replay buffer with prioritized experience replay"""
    def __init__(self, buffer_size, batch_size, n_step=3, gamma=0.99, alpha=0.6, beta=0.4):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.pos = 0
        self.buffer_size = buffer_size
        
        # For N-step learning
        self.n_step_buffer = deque(maxlen=n_step)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Add to N-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If we don't have enough experiences yet, don't add to main buffer
        if len(self.n_step_buffer) < self.n_step and not done:
            return
        
        # Calculate N-step return
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        # Calculate discounted return
        reward = 0
        for i in range(len(self.n_step_buffer)):
            r = self.n_step_buffer[i][2]
            reward += r * (self.gamma ** i)
        
        # Add to buffer with max priority
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done, self.max_priority)
        self.pos = (self.pos + 1) % self.buffer_size
        
        # Remove the oldest experience from N-step buffer
        if not done:
            self.n_step_buffer.popleft()
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if len(self.buffer) < self.batch_size or len(self.buffer) == 0:
            return None
            
        try:
            # Calculate sampling probabilities
            priorities = np.array([e[5] for e in self.buffer if e is not None])
            probs = priorities ** self.alpha
            probs_sum = probs.sum()
            
            # Handle case where all priorities are zero
            if probs_sum == 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs_sum
            
            # Sample indices based on probabilities
            indices = np.random.choice(len(self.buffer), min(self.batch_size, len(self.buffer)), p=probs, replace=False)
            
            # Get samples
            samples = [self.buffer[idx] for idx in indices]
            states, actions, rewards, next_states, dones, _ = zip(*samples)
            
            # Calculate importance-sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            if len(weights) > 0:
                weights = weights / weights.max()
            
            return (np.array(states, dtype=np.float32), 
                   np.array(actions, dtype=np.int64), 
                   np.array(rewards, dtype=np.float32), 
                   np.array(next_states, dtype=np.float32), 
                   np.array(dones, dtype=np.float32), 
                   indices, 
                   np.array(weights, dtype=np.float32))
        except Exception as e:
            print(f"Error in sample(): {e}")
            return None
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.max_priority = max(self.max_priority, priority)
            self.buffer[idx] = (*self.buffer[idx][:-1], priority)
    
    def __len__(self):
        return len(self.buffer)

class AdvancedAgent(BaseAgentPolicy):
    """Advanced agent with state-of-the-art RL techniques."""
    
    def __init__(
        self,
        agent_id: str,
        env: PyQuaticusEnv,  # We know we're only using PyQuaticusEnv
        state_size: int = 20,  # Adjust based on your state representation
        action_size: int = 5,  # Adjust based on your action space
        seed: int = 0,
        lr: float = LR,
        gamma: float = GAMMA,
        tau: float = TAU,
        update_every: int = UPDATE_EVERY,
        learn_num: int = LEARN_NUM,
        batch_size: int = BATCH_SIZE,
        buffer_size: int = BUFFER_SIZE,
        n_step: int = N_STEPS,
        alpha: float = ALPHA,
        beta: float = BETA,
        device: torch.device = device
    ):
        """Initialize an AdvancedAgent object.
        
        Args:
            agent_id: Agent identifier
            env: Environment instance
            state_size: Dimension of state space
            action_size: Dimension of action space
            seed: Random seed
            lr: Learning rate
            gamma: Discount factor
            tau: For soft update of target parameters
            update_every: How often to update the network
            learn_num: Number of learning updates per step
            batch_size: Minibatch size
            buffer_size: Replay buffer size
            n_step: Number of steps for N-step learning
            alpha: Priority exponent
            beta: Importance sampling exponent
            device: Device to run the model on (CPU/GPU)
        """
        # Initialize the parent class with agent_id and env
        super().__init__(agent_id, env)
        
        # Store the agent_id as an instance variable
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learn_num = learn_num
        self.batch_size = batch_size
        self.device = device
        
        # Q-Network
        self.qnetwork_local = DuelingNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingNetwork(state_size, action_size, seed).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200000)
        
        # Replay memory
        self.memory = MultiStepBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_step=n_step,
            gamma=gamma,
            alpha=alpha,
            beta=beta
        )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Epsilon-greedy parameters
        self.eps = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        
        # For tracking training progress
        self.losses = []
        self.rewards = []
        self.episode_reward = 0
        
        # Initialize target network
        self.soft_update(1.0)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                for _ in range(self.learn_num):
                    try:
                        experiences = self.memory.sample()
                        # Check if we got valid experiences
                        if experiences is not None and len(experiences) == 7:  # 7 elements: states, actions, rewards, next_states, dones, indices, weights
                            self.learn(experiences)
                    except Exception as e:
                        print(f"Warning: Error during experience sampling: {e}")
                        continue
    
    def act(self, state, eps=None):
        """Returns actions for given state as per current policy.
        
        Args:
            state: Current state
            eps: Epsilon, for epsilon-greedy action selection
            
        Returns:
            Action index
        """
        if eps is None:
            eps = self.eps
            
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        if experiences is None or len(experiences) < 2:  # Skip learning if not enough samples
            return 0.0
            
        try:
            states, actions, rewards, next_states, dones, indices, weights = experiences
            
            # Convert to PyTorch tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
            
            # Double DQN target: online net selects action, target net evaluates it
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Compute Q targets for current states 
                Q_expected = self.qnetwork_local(states).gather(1, actions)
                # Get max predicted Q values (for next states) from target model
                Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
                # Compute Q targets for current states 
                Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            
            # Compute loss with importance sampling weights
            loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
            
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
            
            self.optimizer.step()
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()
            
            # Update target network
            self.soft_update(self.tau)
            
            # Reset noise in noisy layers
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()
            
            # Update priorities in replay buffer
            with torch.no_grad():
                td_errors = (Q_expected - Q_targets).abs().squeeze().cpu().numpy()
                self.memory.update_priorities(indices, td_errors + 1e-5)  # Small constant to avoid zero priority
            
            # Update epsilon
            self.eps = max(self.eps_end, self.eps_decay * self.eps)
            
            # Track loss
            self.losses.append(loss.item())
            
        except Exception as e:
            print(f"Error in learn(): {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        
        self.optimizer.step()
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step()
        
        # Update target network
        self.soft_update(self.tau)
        
        # Reset noise in noisy layers
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
        
        # Update priorities in replay buffer
        with torch.no_grad():
            td_errors = (Q_expected - Q_targets).abs().squeeze().cpu().numpy()
            self.memory.update_priorities(indices, td_errors + PRIO_EPSILON)
        
        # Update epsilon
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        
        # Track loss
        self.losses.append(loss.item())
    
    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, filename):
        """Save the model weights to a file."""
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eps': self.eps,
            'losses': self.losses,
            'rewards': self.rewards
        }, filename)
    
    def load(self, filename):
        """Load model weights from a file."""
        if os.path.isfile(filename):
            print(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.eps = checkpoint['eps']
            self.losses = checkpoint['losses']
            self.rewards = checkpoint['rewards']
            
            print(f"Loaded checkpoint '{filename}' (eps: {self.eps:.4f})")
        else:
            print(f"No checkpoint found at '{filename}'")
