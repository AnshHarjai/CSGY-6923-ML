"""
Multi-Agent Taxi Environment - PettingZoo Implementation
=========================================================

This module implements a custom multi-agent taxi environment using the PettingZoo API.
Multiple taxi agents operate in a shared grid world, either competing or cooperating
to pick up and drop off passengers.

Features:
- Multiple independent taxi agents
- Stochastic action execution (optional)
- Competitive and cooperative reward schemes
- Full PettingZoo Parallel API compliance

Author: Environment Design Team
Date: November 12, 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import functools
from gymnasium import spaces

# PettingZoo imports
try:
    from pettingzoo import ParallelEnv
    from pettingzoo.utils import parallel_to_aec, wrappers
except ImportError:
    print("Warning: PettingZoo not installed. Install with: pip install pettingzoo")
    ParallelEnv = object


class MultiAgentTaxiEnv(ParallelEnv):
    """
    Multi-agent taxi environment following PettingZoo Parallel API.
    
    The environment simulates multiple taxi agents operating in a 5x5 grid world.
    Each agent must pick up passengers and drop them at their destinations.
    
    Parameters
    ----------
    num_agents : int
        Number of taxi agents (default: 2)
    num_passengers : int
        Number of passengers in the environment (default: 2)
    cooperative : bool
        If True, agents share rewards. If False, agents compete (default: True)
    slip_prob : float
        Probability of stochastic action failure (default: 0.0)
    max_steps : int
        Maximum steps per episode (default: 200)
    
    Observation Space (per agent)
    ------------------------------
    Discrete space with encoded state:
    - Agent row (0-4)
    - Agent column (0-4)
    - Passenger locations for each passenger (0-4 or 5 if in taxi)
    - Destination locations for each passenger (0-3)
    - Other agents' positions (encoded)
    
    Action Space
    ------------
    Discrete(6):
    - 0: Move south
    - 1: Move north
    - 2: Move east
    - 3: Move west
    - 4: Pickup passenger
    - 5: Drop off passenger
    
    Rewards
    -------
    Cooperative mode:
    - All agents share the same reward
    - +20 for successful dropoff
    - -10 for illegal pickup/dropoff
    - -1 per timestep
    
    Competitive mode:
    - Individual rewards per agent
    - +20 for successful dropoff by that agent
    - -10 for illegal actions
    - -1 per timestep
    """
    
    metadata = {
        'render_modes': ['human', 'ansi'],
        'name': 'multiagent_taxi_v0',
        'is_parallelizable': True,
    }
    
    def __init__(
        self,
        num_agents: int = 2,
        num_passengers: int = 2,
        cooperative: bool = True,
        slip_prob: float = 0.0,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        """Initialize multi-agent taxi environment."""
        super().__init__()
        
        # Environment parameters (use underscore for internal storage to avoid property conflicts)
        self._num_agents = num_agents
        self._num_passengers = num_passengers
        self.cooperative = cooperative
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Grid dimensions (matching Taxi-v3)
        self.grid_height = 5
        self.grid_width = 5
        
        # Passenger/destination locations (R, G, Y, B)
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        
        # Agent identifiers
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        
        # Random number generator
        self.rng = np.random.RandomState()
        
        # Initialize state
        self.agents = self.possible_agents[:]
        self._agent_positions = {}
        self._passenger_locations = []
        self._passenger_destinations = []
        self._passenger_in_taxi = {}  # Maps passenger_id -> agent_id or None
        self.steps = 0
    
    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._num_agents
    
    @property
    def num_passengers(self) -> int:
        """Return the number of passengers in the environment."""
        return self._num_passengers
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """
        Return observation space for given agent.
        
        Observation includes:
        - Agent's position (row, col)
        - All passenger locations and destinations
        - Other agents' positions
        """
        # Calculate state space size
        # For simplicity, we use a flattened discrete space
        # More sophisticated: use Dict or MultiDiscrete space
        
        # Components:
        # - Agent position: 25 possibilities (5x5 grid)
        # - Each passenger: 6 locations (5 locs + in taxi)
        # - Each destination: 4 locations
        # - Other agent positions: 25 each
        
        max_state = (
            self.grid_height * self.grid_width *  # This agent's position
            (len(self.locs) + 1) ** self._num_passengers *  # Passenger locations
            len(self.locs) ** self._num_passengers *  # Destinations
            (self.grid_height * self.grid_width) ** (self._num_agents - 1)  # Other agents
        )
        
        return spaces.Discrete(max_state)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for given agent."""
        return spaces.Discrete(6)  # 4 moves + pickup + dropoff
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, int], Dict[str, Dict]]:
        """
        Reset the environment to initial state.
        
        Returns:
            observations: Dict mapping agent_id to observation
            infos: Dict mapping agent_id to info dict
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        self.agents = self.possible_agents[:]
        self.steps = 0
        
        # Initialize agent positions (random non-overlapping positions)
        self._agent_positions = {}
        used_positions = set()
        
        for agent in self.agents:
            while True:
                pos = (self.rng.randint(0, self.grid_height), 
                       self.rng.randint(0, self.grid_width))
                if pos not in used_positions:
                    self._agent_positions[agent] = pos
                    used_positions.add(pos)
                    break
        
        # Initialize passengers
        self._passenger_locations = []
        self._passenger_destinations = []
        self._passenger_in_taxi = {}
        
        for i in range(self._num_passengers):
            # Random start and destination (different)
            start = self.rng.choice(len(self.locs))
            dest = self.rng.choice(len(self.locs))
            while dest == start:
                dest = self.rng.choice(len(self.locs))
            
            self._passenger_locations.append(start)
            self._passenger_destinations.append(dest)
            self._passenger_in_taxi[i] = None  # Not in any taxi
        
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(
        self, 
        actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, int],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict]
    ]:
        """
        Execute one step with actions from all agents.
        
        Args:
            actions: Dict mapping agent_id to action
            
        Returns:
            observations: Dict of observations
            rewards: Dict of rewards
            terminations: Dict of termination flags
            truncations: Dict of truncation flags
            infos: Dict of info dicts
        """
        self.steps += 1
        
        # Process actions with optional stochasticity
        actual_actions = {}
        for agent, action in actions.items():
            if self.rng.random() < self.slip_prob:
                # Action slips
                actual_actions[agent] = self.rng.randint(0, 6)
            else:
                actual_actions[agent] = action
        
        # Execute actions and collect rewards
        agent_rewards = {agent: -1 for agent in self.agents}  # Default: -1 per step
        
        for agent, action in actual_actions.items():
            reward = self._execute_action(agent, action)
            agent_rewards[agent] += reward
        
        # Check if episode is done (all passengers delivered)
        all_delivered = all(
            self._passenger_locations[i] == self._passenger_destinations[i]
            for i in range(self._num_passengers)
        )
        
        truncated = self.steps >= self.max_steps
        terminated = all_delivered
        
        # In cooperative mode, average rewards
        if self.cooperative:
            avg_reward = sum(agent_rewards.values()) / len(agent_rewards)
            rewards = {agent: avg_reward for agent in self.agents}
        else:
            rewards = agent_rewards
        
        # Get observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {'actual_action': actual_actions[agent]} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _execute_action(self, agent: str, action: int) -> float:
        """
        Execute a single agent's action and return reward.
        
        Args:
            agent: Agent identifier
            action: Action to execute (0-5)
            
        Returns:
            reward: Immediate reward for this action
        """
        row, col = self._agent_positions[agent]
        reward = 0
        
        # Movement actions (0-3)
        if action == 0:  # South
            row = min(row + 1, self.grid_height - 1)
        elif action == 1:  # North
            row = max(row - 1, 0)
        elif action == 2:  # East
            col = min(col + 1, self.grid_width - 1)
        elif action == 3:  # West
            col = max(col - 1, 0)
        
        # Update position
        self._agent_positions[agent] = (row, col)
        
        # Pickup action (4)
        if action == 4:
            # Check if there's a passenger at this location
            agent_pos = (row, col)
            pickup_success = False
            
            for i, loc_idx in enumerate(self._passenger_locations):
                if (self._passenger_in_taxi[i] is None and 
                    self.locs[loc_idx] == agent_pos):
                    # Pickup passenger
                    self._passenger_in_taxi[i] = agent
                    self._passenger_locations[i] = len(self.locs)  # Mark as in taxi
                    pickup_success = True
                    break
            
            if not pickup_success:
                reward = -10  # Illegal pickup
        
        # Dropoff action (5)
        elif action == 5:
            agent_pos = (row, col)
            dropoff_success = False
            
            # Check if this agent has a passenger
            for i, carrier in self._passenger_in_taxi.items():
                if carrier == agent:
                    # Check if at correct destination
                    dest_idx = self._passenger_destinations[i]
                    if self.locs[dest_idx] == agent_pos:
                        # Successful dropoff
                        self._passenger_locations[i] = dest_idx
                        self._passenger_in_taxi[i] = None
                        reward = 20
                        dropoff_success = True
                        break
            
            if not dropoff_success:
                reward = -10  # Illegal dropoff
        
        return reward
    
    def _get_observation(self, agent: str) -> int:
        """
        Get observation for a specific agent.
        
        Returns encoded state as integer.
        """
        # Encode state as integer
        # This is a simplified encoding; you might want to use MultiDiscrete
        
        row, col = self._agent_positions[agent]
        state = row * self.grid_width + col
        
        # Encode passenger information
        for i in range(self._num_passengers):
            pass_loc = self._passenger_locations[i]
            pass_dest = self._passenger_destinations[i]
            state = state * (len(self.locs) + 1) + pass_loc
            state = state * len(self.locs) + pass_dest
        
        # Encode other agents' positions
        for other_agent in self.agents:
            if other_agent != agent:
                other_row, other_col = self._agent_positions[other_agent]
                state = state * self.grid_height * self.grid_width + (other_row * self.grid_width + other_col)
        
        return state
    
    def render(self):
        """Render the environment (text-based)."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            return self._render_text()
    
    def _render_text(self) -> str:
        """Create text representation of environment."""
        grid = [['.' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Mark special locations
        for idx, (r, c) in enumerate(self.locs):
            grid[r][c] = ['R', 'G', 'Y', 'B'][idx]
        
        # Place agents
        for agent_idx, agent in enumerate(self.agents):
            r, c = self._agent_positions[agent]
            grid[r][c] = str(agent_idx)
        
        # Create string
        output = [f"Step: {self.steps}/{self.max_steps}"]
        output.append("+" + "-" * self.grid_width + "+")
        for row in grid:
            output.append("|" + "".join(row) + "|")
        output.append("+" + "-" * self.grid_width + "+")
        
        # Add passenger info
        for i in range(self._num_passengers):
            loc = self._passenger_locations[i]
            dest = self._passenger_destinations[i]
            carrier = self._passenger_in_taxi[i]
            
            if carrier:
                output.append(f"Passenger {i}: In {carrier}, Destination: {['R','G','Y','B'][dest]}")
            else:
                output.append(f"Passenger {i}: At {['R','G','Y','B'][loc]}, Destination: {['R','G','Y','B'][dest]}")
        
        result = "\n".join(output)
        
        if self.render_mode == "human":
            print(result)
        
        return result
    
    def close(self):
        """Clean up environment resources."""
        pass


def make_multiagent_taxi(
    num_agents: int = 2,
    num_passengers: int = 2,
    cooperative: bool = True,
    slip_prob: float = 0.0,
    max_steps: int = 200,
) -> MultiAgentTaxiEnv:
    """
    Convenience function to create multi-agent taxi environment.
    
    Args:
        num_agents: Number of taxi agents
        num_passengers: Number of passengers
        cooperative: Whether agents cooperate or compete
        slip_prob: Probability of stochastic action failure
        max_steps: Maximum steps per episode
        
    Returns:
        MultiAgentTaxiEnv instance
    """
    return MultiAgentTaxiEnv(
        num_agents=num_agents,
        num_passengers=num_passengers,
        cooperative=cooperative,
        slip_prob=slip_prob,
        max_steps=max_steps,
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Taxi Environment - Demo")
    print("=" * 60)
    
    # Test 1: Cooperative mode
    print("\n--- Test 1: Cooperative Mode (2 agents, 2 passengers) ---")
    env = make_multiagent_taxi(num_agents=2, num_passengers=2, cooperative=True, slip_prob=0.1)
    
    observations, infos = env.reset(seed=42)
    print(f"Agents: {env.agents}")
    print(f"Initial observations: {observations}")
    
    # Run a few steps with random actions
    for step in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if step < 3:  # Print first 3 steps
            print(f"\nStep {step + 1}:")
            print(f"  Actions: {actions}")
            print(f"  Rewards: {rewards}")
            print(f"  Terminated: {terminations}")
        
        if all(terminations.values()) or all(truncations.values()):
            print(f"\nEpisode ended at step {step + 1}")
            break
    
    # Test 2: Competitive mode
    print("\n\n--- Test 2: Competitive Mode (3 agents, 3 passengers) ---")
    env = make_multiagent_taxi(num_agents=3, num_passengers=3, cooperative=False, slip_prob=0.0)
    
    observations, infos = env.reset(seed=123)
    print(f"Agents: {env.agents}")
    print(f"Observation space: {env.observation_space(env.agents[0])}")
    print(f"Action space: {env.action_space(env.agents[0])}")
    
    # Test 3: Rendering
    print("\n\n--- Test 3: Environment Rendering ---")
    env = make_multiagent_taxi(num_agents=2, num_passengers=1, cooperative=True)
    env.render_mode = "ansi"
    observations, infos = env.reset(seed=999)
    
    print(env.render())
    
    # Take some actions
    actions = {agent: 0 for agent in env.agents}  # All move south
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    print("\nAfter all agents move south:")
    print(env.render())
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
