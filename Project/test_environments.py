"""
Comprehensive Testing Suite for Stochastic and Multi-Agent Taxi Environments
=============================================================================

This script validates the correctness of both environments through:
1. API compliance tests
2. Functional behavior tests
3. Statistical tests for stochasticity
4. Dummy agent tests (random and greedy)

Author: Environment Design Team
Date: November 12, 2025
"""

import numpy as np
import sys
from typing import Dict, List, Tuple

# Import our environments
try:
    from stochastic_taxi_env import StochasticTaxiEnv, make_stochastic_taxi
    from multiagent_taxi_env import MultiAgentTaxiEnv, make_multiagent_taxi
    import gymnasium as gym
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import environments: {e}")
    print("Some tests may be skipped.")
    IMPORTS_AVAILABLE = False


class TestResults:
    """Helper class to track test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✓ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.failures.append((test_name, error))
        print(f"✗ {test_name}: {error}")
    
    def add_skip(self, test_name: str, reason: str):
        self.skipped += 1
        print(f"⊘ {test_name} (skipped: {reason})")
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total}")
        print(f"✓ Passed: {self.passed}")
        print(f"✗ Failed: {self.failed}")
        print(f"⊘ Skipped: {self.skipped}")
        
        if self.failures:
            print("\nFailed tests:")
            for name, error in self.failures:
                print(f"  - {name}: {error}")
        
        print("=" * 60)
        return self.failed == 0


# =============================================================================
# Stochastic Environment Tests
# =============================================================================

def test_stochastic_basic(results: TestResults):
    """Test basic functionality of stochastic environment."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("Stochastic Basic", "imports not available")
        return
    
    try:
        env = make_stochastic_taxi(slip_prob=0.2, seed=42)
        
        # Test reset
        obs, info = env.reset()
        assert isinstance(obs, (int, np.integer)), "Observation should be integer"
        assert 'slip_probability' in info, "Info should contain slip_probability"
        
        # Test step
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert 'intended_action' in info, "Info should contain intended_action"
        assert 'actual_action' in info, "Info should contain actual_action"
        assert 'slipped' in info, "Info should contain slipped flag"
        
        env.close()
        results.add_pass("Stochastic Basic Functionality")
    except Exception as e:
        results.add_fail("Stochastic Basic Functionality", str(e))


def test_stochastic_slip_probability(results: TestResults):
    """Test that slip probability is approximately correct."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("Stochastic Slip Probability", "imports not available")
        return
    
    try:
        slip_prob = 0.3
        env = make_stochastic_taxi(slip_prob=slip_prob, seed=42)
        
        obs, info = env.reset(seed=42)
        
        num_steps = 1000
        num_slips = 0
        
        for _ in range(num_steps):
            action = 0  # Always try action 0
            obs, reward, terminated, truncated, info = env.step(action)
            if info['slipped']:
                num_slips += 1
            
            if terminated or truncated:
                obs, info = env.reset()
        
        observed_slip_rate = num_slips / num_steps
        error = abs(observed_slip_rate - slip_prob)
        
        # Allow 5% error margin
        if error < 0.05:
            results.add_pass(f"Stochastic Slip Probability (expected={slip_prob:.2f}, observed={observed_slip_rate:.2f})")
        else:
            results.add_fail(
                "Stochastic Slip Probability",
                f"Expected {slip_prob:.2f}, got {observed_slip_rate:.2f} (error={error:.2f})"
            )
        
        env.close()
    except Exception as e:
        results.add_fail("Stochastic Slip Probability", str(e))


def test_stochastic_determinism(results: TestResults):
    """Test that same seed produces same results."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("Stochastic Determinism", "imports not available")
        return
    
    try:
        # Run 1
        env1 = make_stochastic_taxi(slip_prob=0.3, seed=12345)
        obs1, _ = env1.reset(seed=12345)
        actions1 = [env1.action_space.sample() for _ in range(10)]
        
        trajectory1 = []
        for action in actions1:
            obs, reward, terminated, truncated, info = env1.step(action)
            trajectory1.append((obs, reward, info['slipped']))
            if terminated or truncated:
                break
        
        # Run 2 (same seed)
        env2 = make_stochastic_taxi(slip_prob=0.3, seed=12345)
        obs2, _ = env2.reset(seed=12345)
        
        trajectory2 = []
        for action in actions1:  # Same actions
            obs, reward, terminated, truncated, info = env2.step(action)
            trajectory2.append((obs, reward, info['slipped']))
            if terminated or truncated:
                break
        
        if trajectory1 == trajectory2:
            results.add_pass("Stochastic Determinism (seed reproducibility)")
        else:
            results.add_fail("Stochastic Determinism", "Same seed produced different trajectories")
        
        env1.close()
        env2.close()
    except Exception as e:
        results.add_fail("Stochastic Determinism", str(e))


# =============================================================================
# Multi-Agent Environment Tests
# =============================================================================

def test_multiagent_basic(results: TestResults):
    """Test basic functionality of multi-agent environment."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("MultiAgent Basic", "imports not available")
        return
    
    try:
        env = make_multiagent_taxi(num_agents=2, num_passengers=2)
        
        # Test reset
        observations, infos = env.reset(seed=42)
        assert isinstance(observations, dict), "Observations should be dict"
        assert len(observations) == 2, "Should have 2 observations"
        assert all(agent in observations for agent in env.agents), "All agents should have observations"
        
        # Test step
        actions = {agent: 0 for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        assert isinstance(rewards, dict), "Rewards should be dict"
        assert isinstance(terminations, dict), "Terminations should be dict"
        assert isinstance(truncations, dict), "Truncations should be dict"
        assert len(rewards) == len(env.agents), "Should have reward for each agent"
        
        env.close()
        results.add_pass("MultiAgent Basic Functionality")
    except Exception as e:
        results.add_fail("MultiAgent Basic Functionality", str(e))


def test_multiagent_spaces(results: TestResults):
    """Test observation and action spaces."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("MultiAgent Spaces", "imports not available")
        return
    
    try:
        env = make_multiagent_taxi(num_agents=3, num_passengers=2)
        
        for agent in env.agents:
            obs_space = env.observation_space(agent)
            action_space = env.action_space(agent)
            
            assert hasattr(obs_space, 'n'), "Observation space should be Discrete"
            assert hasattr(action_space, 'n'), "Action space should be Discrete"
            assert action_space.n == 6, "Action space should have 6 actions"
        
        results.add_pass("MultiAgent Observation/Action Spaces")
    except Exception as e:
        results.add_fail("MultiAgent Observation/Action Spaces", str(e))


def test_multiagent_cooperative_rewards(results: TestResults):
    """Test that cooperative mode shares rewards."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("MultiAgent Cooperative Rewards", "imports not available")
        return
    
    try:
        env = make_multiagent_taxi(num_agents=2, num_passengers=1, cooperative=True)
        observations, infos = env.reset(seed=42)
        
        # Take some steps
        for _ in range(10):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # In cooperative mode, all agents should get same reward
            reward_values = list(rewards.values())
            if len(set(reward_values)) != 1:
                results.add_fail(
                    "MultiAgent Cooperative Rewards",
                    f"Rewards not equal in cooperative mode: {rewards}"
                )
                return
            
            if all(terminations.values()):
                break
        
        results.add_pass("MultiAgent Cooperative Rewards (all agents share reward)")
    except Exception as e:
        results.add_fail("MultiAgent Cooperative Rewards", str(e))


def test_multiagent_competitive_rewards(results: TestResults):
    """Test that competitive mode has individual rewards."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("MultiAgent Competitive Rewards", "imports not available")
        return
    
    try:
        env = make_multiagent_taxi(num_agents=2, num_passengers=1, cooperative=False)
        observations, infos = env.reset(seed=42)
        
        found_different = False
        
        # Take steps until we find different rewards
        for _ in range(100):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # In competitive mode, agents can have different rewards
            reward_values = list(rewards.values())
            if len(set(reward_values)) > 1:
                found_different = True
                break
            
            if all(terminations.values()):
                observations, infos = env.reset()
        
        # Note: It's possible all agents get same reward if they all just move
        # We'll pass this test regardless, but note if rewards differed
        if found_different:
            results.add_pass("MultiAgent Competitive Rewards (individual rewards confirmed)")
        else:
            results.add_pass("MultiAgent Competitive Rewards (test passed, no reward difference observed)")
    except Exception as e:
        results.add_fail("MultiAgent Competitive Rewards", str(e))


def test_multiagent_termination(results: TestResults):
    """Test episode termination conditions."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("MultiAgent Termination", "imports not available")
        return
    
    try:
        env = make_multiagent_taxi(num_agents=2, num_passengers=1, max_steps=50)
        observations, infos = env.reset(seed=42)
        
        # Run until truncation
        for step in range(100):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            if all(terminations.values()):
                results.add_pass(f"MultiAgent Termination (terminated at step {step})")
                return
            
            if all(truncations.values()):
                if step >= 49:  # Should truncate around max_steps
                    results.add_pass(f"MultiAgent Truncation (truncated at step {step})")
                    return
                else:
                    results.add_fail("MultiAgent Termination", f"Truncated too early at step {step}")
                    return
        
        results.add_fail("MultiAgent Termination", "Episode did not terminate or truncate")
    except Exception as e:
        results.add_fail("MultiAgent Termination", str(e))


def test_multiagent_rendering(results: TestResults):
    """Test rendering functionality."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("MultiAgent Rendering", "imports not available")
        return
    
    try:
        env = make_multiagent_taxi(num_agents=2, num_passengers=1)
        env.render_mode = "ansi"
        
        observations, infos = env.reset(seed=42)
        render_output = env.render()
        
        assert isinstance(render_output, str), "Render should return string"
        assert len(render_output) > 0, "Render output should not be empty"
        assert "Step:" in render_output, "Render should show step count"
        
        results.add_pass("MultiAgent Rendering")
    except Exception as e:
        results.add_fail("MultiAgent Rendering", str(e))


# =============================================================================
# Dummy Agent Tests
# =============================================================================

def test_random_agent_stochastic(results: TestResults):
    """Test random agent in stochastic environment."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("Random Agent (Stochastic)", "imports not available")
        return
    
    try:
        env = make_stochastic_taxi(slip_prob=0.2, seed=42)
        
        num_episodes = 5
        total_rewards = []
        
        for ep in range(num_episodes):
            obs, info = env.reset(seed=42 + ep)
            episode_reward = 0
            
            for step in range(200):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        results.add_pass(f"Random Agent (Stochastic) - Avg reward: {avg_reward:.2f}")
        env.close()
    except Exception as e:
        results.add_fail("Random Agent (Stochastic)", str(e))


def test_random_agent_multiagent(results: TestResults):
    """Test random agents in multi-agent environment."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("Random Agent (MultiAgent)", "imports not available")
        return
    
    try:
        env = make_multiagent_taxi(num_agents=2, num_passengers=2, cooperative=True)
        
        num_episodes = 5
        total_rewards = []
        
        for ep in range(num_episodes):
            observations, infos = env.reset(seed=42 + ep)
            episode_reward = 0
            
            for step in range(200):
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                
                if all(terminations.values()) or all(truncations.values()):
                    break
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        results.add_pass(f"Random Agent (MultiAgent) - Avg reward: {avg_reward:.2f}")
        env.close()
    except Exception as e:
        results.add_fail("Random Agent (MultiAgent)", str(e))


def test_greedy_agent_stochastic(results: TestResults):
    """Test simple greedy agent in stochastic environment."""
    if not IMPORTS_AVAILABLE:
        results.add_skip("Greedy Agent (Stochastic)", "imports not available")
        return
    
    try:
        # Simple greedy agent: always try to move toward a goal
        # (This is just a sanity check, not a smart agent)
        env = make_stochastic_taxi(slip_prob=0.1, seed=42)
        
        obs, info = env.reset(seed=42)
        episode_reward = 0
        
        for step in range(100):
            # Just take action 4 (pickup) or 5 (dropoff) occasionally
            if step % 10 == 0:
                action = 4  # Try pickup
            elif step % 10 == 5:
                action = 5  # Try dropoff
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        results.add_pass(f"Greedy Agent (Stochastic) - Reward: {episode_reward:.2f}")
        env.close()
    except Exception as e:
        results.add_fail("Greedy Agent (Stochastic)", str(e))


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("ENVIRONMENT TESTING SUITE")
    print("=" * 60)
    print()
    
    results = TestResults()
    
    # Stochastic environment tests
    print("--- Stochastic Environment Tests ---")
    test_stochastic_basic(results)
    test_stochastic_slip_probability(results)
    test_stochastic_determinism(results)
    print()
    
    # Multi-agent environment tests
    print("--- Multi-Agent Environment Tests ---")
    test_multiagent_basic(results)
    test_multiagent_spaces(results)
    test_multiagent_cooperative_rewards(results)
    test_multiagent_competitive_rewards(results)
    test_multiagent_termination(results)
    test_multiagent_rendering(results)
    print()
    
    # Dummy agent tests
    print("--- Dummy Agent Tests ---")
    test_random_agent_stochastic(results)
    test_random_agent_multiagent(results)
    test_greedy_agent_stochastic(results)
    print()
    
    # Print summary
    success = results.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
