"""
Stochastic Taxi Environment - Gymnasium Wrapper
================================================

This module extends the Taxi-v3 environment by introducing stochastic action failures
(slip probability). When an action is taken, there's a probability that the action will
fail and a random action will be executed instead.

Author: Environment Design Team
Date: November 12, 2025
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class StochasticTaxiEnv(gym.Wrapper):
    """
    A wrapper for the Taxi-v3 environment that introduces stochastic action failures.
    
    With probability `slip_prob`, the intended action will fail and a random action
    will be executed instead. This simulates real-world uncertainty in action execution.
    
    Parameters
    ----------
    env : gym.Env
        The base Taxi-v3 environment to wrap
    slip_prob : float
        Probability of action failure (default: 0.2)
    seed : Optional[int]
        Random seed for reproducibility
    
    Example
    -------
    >>> import gymnasium as gym
    >>> base_env = gym.make('Taxi-v3')
    >>> stochastic_env = StochasticTaxiEnv(base_env, slip_prob=0.2)
    >>> obs, info = stochastic_env.reset()
    >>> obs, reward, terminated, truncated, info = stochastic_env.step(0)
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        slip_prob: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize the stochastic taxi environment wrapper.
        
        Args:
            env: Base Gymnasium environment (typically Taxi-v3)
            slip_prob: Probability that an action will slip (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError(f"slip_prob must be between 0 and 1, got {slip_prob}")
        
        self.slip_prob = slip_prob
        self.rng = np.random.RandomState(seed)
        
        # Store action space size for random action selection
        self.num_actions = self.action_space.n
        
        # Metadata
        self.metadata = {
            **env.metadata,
            'stochastic': True,
            'slip_probability': slip_prob
        }
    
    def step(
        self, 
        action: int
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with stochastic action execution.
        
        Args:
            action: The intended action to take
            
        Returns:
            observation: The resulting state
            reward: Reward received
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated
            info: Additional information dictionary
        """
        # Determine if action slips
        actual_action = action
        slipped = False
        
        if self.rng.random() < self.slip_prob:
            # Action fails - execute random action instead
            actual_action = self.rng.randint(0, self.num_actions)
            slipped = True
        
        # Execute the actual action
        observation, reward, terminated, truncated, info = self.env.step(actual_action)
        
        # Add slip information to info dict
        info['intended_action'] = action
        info['actual_action'] = actual_action
        info['slipped'] = slipped
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Information dictionary
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        observation, info = self.env.reset(seed=seed, options=options)
        info['slip_probability'] = self.slip_prob
        
        return observation, info
    
    def set_slip_prob(self, slip_prob: float):
        """
        Update the slip probability.
        
        Args:
            slip_prob: New slip probability (0.0 to 1.0)
        """
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError(f"slip_prob must be between 0 and 1, got {slip_prob}")
        self.slip_prob = slip_prob


def make_stochastic_taxi(slip_prob: float = 0.2, seed: Optional[int] = None) -> StochasticTaxiEnv:
    """
    Convenience function to create a stochastic taxi environment.
    
    Args:
        slip_prob: Probability of action failure
        seed: Random seed for reproducibility
        
    Returns:
        StochasticTaxiEnv instance
    """
    base_env = gym.make('Taxi-v3')
    return StochasticTaxiEnv(base_env, slip_prob=slip_prob, seed=seed)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Stochastic Taxi Environment - Demo")
    print("=" * 60)
    
    # Create stochastic environment
    env = make_stochastic_taxi(slip_prob=0.3, seed=42)
    
    print(f"\nEnvironment created with slip probability: {env.slip_prob}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few episodes
    num_episodes = 3
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial observation: {obs}")
        
        total_reward = 0
        slips = 0
        steps = 0
        
        for step in range(50):  # Max 50 steps per episode
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if info['slipped']:
                slips += 1
                print(f"  Step {step}: Action {info['intended_action']} slipped to {info['actual_action']}")
            
            if terminated or truncated:
                break
        
        print(f"Episode finished in {steps} steps")
        print(f"Total reward: {total_reward}")
        print(f"Slips: {slips}/{steps} ({100*slips/steps:.1f}%)")
    
    env.close()
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
