"""
Demo Script - Stochastic and Multi-Agent Taxi Environments
===========================================================

This script demonstrates the usage of both environments with various
configurations and scenarios.

Author: Environment Design Team
Date: November 12, 2025
"""

import numpy as np
import time

try:
    from stochastic_taxi_env import make_stochastic_taxi
    from multiagent_taxi_env import make_multiagent_taxi
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing environments: {e}")
    print("\nPlease install required packages:")
    print("  pip install gymnasium pettingzoo numpy")
    IMPORTS_AVAILABLE = False


def demo_stochastic_taxi():
    """Demonstrate stochastic taxi environment."""
    print("=" * 70)
    print("DEMO 1: Stochastic Taxi Environment")
    print("=" * 70)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("Skipping demo - imports not available")
        return
    
    # Create environment with 20% slip probability
    print("Creating environment with 20% slip probability...")
    env = make_stochastic_taxi(slip_prob=0.2, seed=42)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    # Run one episode
    print("Running one episode with random actions...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    
    total_reward = 0
    slips = 0
    steps = 0
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if info['slipped']:
            slips += 1
            if slips <= 3:  # Print first 3 slips
                print(f"  Step {step}: Action {info['intended_action']} → {info['actual_action']} (SLIPPED)")
        
        if terminated:
            print(f"\n✓ Episode completed successfully at step {step}!")
            break
        
        if truncated:
            print(f"\n⊗ Episode truncated at step {step}")
            break
    
    print(f"\nEpisode Statistics:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Slips: {slips}/{steps} ({100*slips/steps:.1f}%)")
    print(f"  Expected slip rate: 20%")
    
    env.close()
    print()


def demo_multiagent_cooperative():
    """Demonstrate cooperative multi-agent environment."""
    print("=" * 70)
    print("DEMO 2: Multi-Agent Taxi - Cooperative Mode")
    print("=" * 70)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("Skipping demo - imports not available")
        return
    
    # Create cooperative environment
    print("Creating cooperative environment (2 agents, 2 passengers)...")
    env = make_multiagent_taxi(
        num_agents=2,
        num_passengers=2,
        cooperative=True,
        slip_prob=0.1,
        max_steps=100
    )
    
    print(f"Agents: {env.agents}")
    print(f"Cooperative mode: Agents share rewards")
    print()
    
    # Run one episode
    print("Running episode with random actions...")
    observations, infos = env.reset(seed=42)
    env.render_mode = "ansi"
    
    print("\nInitial state:")
    print(env.render())
    
    total_rewards = {agent: 0 for agent in env.agents}
    
    for step in range(100):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
        
        # Show state every 20 steps
        if step % 20 == 0 and step > 0:
            print(f"\nState at step {step}:")
            print(env.render())
        
        if all(terminations.values()):
            print(f"\n✓ All passengers delivered at step {step}!")
            print("\nFinal state:")
            print(env.render())
            break
        
        if all(truncations.values()):
            print(f"\n⊗ Episode truncated at step {step}")
            break
    
    print(f"\nFinal Rewards (Cooperative):")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")
    
    env.close()
    print()


def demo_multiagent_competitive():
    """Demonstrate competitive multi-agent environment."""
    print("=" * 70)
    print("DEMO 3: Multi-Agent Taxi - Competitive Mode")
    print("=" * 70)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("Skipping demo - imports not available")
        return
    
    # Create competitive environment
    print("Creating competitive environment (3 agents, 3 passengers)...")
    env = make_multiagent_taxi(
        num_agents=3,
        num_passengers=3,
        cooperative=False,  # Competitive!
        slip_prob=0.0,
        max_steps=150
    )
    
    print(f"Agents: {env.agents}")
    print(f"Competitive mode: Agents compete for passengers")
    print()
    
    # Run one episode
    print("Running episode with random actions...")
    observations, infos = env.reset(seed=123)
    env.render_mode = "ansi"
    
    print("\nInitial state:")
    print(env.render())
    
    total_rewards = {agent: 0 for agent in env.agents}
    deliveries = {agent: 0 for agent in env.agents}
    
    for step in range(150):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            # Count successful deliveries (reward of 20)
            if rewards[agent] >= 20:
                deliveries[agent] += 1
        
        if all(terminations.values()):
            print(f"\n✓ All passengers delivered at step {step}!")
            print("\nFinal state:")
            print(env.render())
            break
        
        if all(truncations.values()):
            print(f"\n⊗ Episode truncated at step {step}")
            break
    
    print(f"\nFinal Results (Competitive):")
    print(f"\nRewards:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")
    
    print(f"\nSuccessful Deliveries:")
    for agent, count in deliveries.items():
        print(f"  {agent}: {count}")
    
    winner = max(total_rewards, key=total_rewards.get)
    print(f"\n🏆 Winner: {winner}")
    
    env.close()
    print()


def demo_stochasticity_comparison():
    """Compare performance across different slip probabilities."""
    print("=" * 70)
    print("DEMO 4: Stochasticity Comparison")
    print("=" * 70)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("Skipping demo - imports not available")
        return
    
    slip_probs = [0.0, 0.1, 0.2, 0.3, 0.5]
    num_episodes = 10
    
    print(f"Testing performance with different slip probabilities...")
    print(f"Running {num_episodes} episodes for each configuration...")
    print()
    
    results = {}
    
    for slip_prob in slip_probs:
        env = make_stochastic_taxi(slip_prob=slip_prob, seed=42)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            obs, info = env.reset(seed=42 + ep)
            episode_reward = 0
            steps = 0
            
            for step in range(200):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        results[slip_prob] = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
        }
        
        env.close()
    
    print("Results:")
    print(f"{'Slip Prob':<12} {'Avg Reward':<15} {'Std Dev':<15} {'Avg Steps':<12}")
    print("-" * 60)
    for slip_prob, stats in results.items():
        print(f"{slip_prob:<12.1f} {stats['avg_reward']:<15.2f} {stats['std_reward']:<15.2f} {stats['avg_length']:<12.1f}")
    
    print()


def demo_scalability():
    """Demonstrate scalability with varying numbers of agents."""
    print("=" * 70)
    print("DEMO 5: Scalability Test")
    print("=" * 70)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("Skipping demo - imports not available")
        return
    
    configs = [
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    ]
    
    print("Testing environment with different numbers of agents...")
    print()
    
    for num_agents, num_passengers in configs:
        env = make_multiagent_taxi(
            num_agents=num_agents,
            num_passengers=num_passengers,
            cooperative=True,
            max_steps=200
        )
        
        # Time reset and step operations
        start = time.time()
        observations, infos = env.reset(seed=42)
        reset_time = time.time() - start
        
        start = time.time()
        for _ in range(100):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            if all(terminations.values()) or all(truncations.values()):
                observations, infos = env.reset()
        step_time = (time.time() - start) / 100
        
        print(f"Agents: {num_agents}, Passengers: {num_passengers}")
        print(f"  Reset time: {reset_time*1000:.2f} ms")
        print(f"  Step time:  {step_time*1000:.2f} ms")
        print()
        
        env.close()


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "ENVIRONMENT DEMONSTRATION SUITE" + " " * 22 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    if not IMPORTS_AVAILABLE:
        print("Please install required packages first:")
        print("  pip install -r requirements.txt")
        return
    
    try:
        demo_stochastic_taxi()
        input("Press Enter to continue to next demo...")
        print("\n")
        
        demo_multiagent_cooperative()
        input("Press Enter to continue to next demo...")
        print("\n")
        
        demo_multiagent_competitive()
        input("Press Enter to continue to next demo...")
        print("\n")
        
        demo_stochasticity_comparison()
        input("Press Enter to continue to next demo...")
        print("\n")
        
        demo_scalability()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 20 + "DEMOS COMPLETED" + " " * 33 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")


if __name__ == "__main__":
    main()
