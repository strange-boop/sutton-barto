import random
import math
import numpy as np
from tqdm import tqdm

# 1. Create N k-armed bandit problems (N=2000, k=10)
# 2. For each problem, set q*(a) for each a according to a normal (Gaussian)
#    distribution with mean 0 and variance 1
# 3. When a learning method performs an action A, select from a normal
#    distribution with mean q*(A) and variance 1


class Agent:

    def __init__(self, explore_fraction, num_actions, default_expected_value):
        self.explore_fraction = explore_fraction
        self.action_value_estimates = np.full(num_actions,
                                              default_expected_value)
        self.rewards_received = []
        self.actions_taken = []
        self.optimal_action = []
        self.num_actions = 
        self.action_counter = {a: 0 for a in range(num_actions)}

    def select_action(self):
        explore = random.random() <= self.explore_fraction
        if explore:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.action_value_estimates)
        return int(action)

    def update_estimated_value(self, action, reward):
        # Naive way
        # rewards_from_action = [
        #     self.rewards_received[i]
        #     for i, a in enumerate(self.actions_taken) if a == action
        # ]
        # new_estimate = sum(rewards_from_action) / len(rewards_from_action)
        # Incremental updates
        old_estimate = self.action_value_estimates[action]
        n = self.action_counter[action]
        new_estimate = old_estimate + (1 / n) * (reward - old_estimate)
        self.action_value_estimates[action] = new_estimate

    def take_action(self, problem):
        action = self.select_action()
        self.actions_taken.append(action)
        self.action_counter[action] += 1
        self.optimal_action.append(action == problem.optimal_action)
        reward = problem.get_reward_for_action(action)
        self.rewards_received.append(reward)
        self.update_estimated_value(action, reward)


class AgentConstantAlpha(Agent):

    def update_estimated_value(self, action, reward):
        old_estimate = self.action_value_estimates[action]
        new_estimate = old_estimate + 0.1 * (reward - old_estimate)
        self.action_value_estimates[action] = new_estimate


class BanditProblem:

    def __init__(self, actions, expected_reward_mean,
                 expected_reward_variance, reward_variance,
                 stationary=True):
        self.action_reward_means = np.ones([actions, 1]) * np.random.normal(
            expected_reward_mean,
            math.sqrt(expected_reward_variance),
            size=[actions, 1]
        )
        self.optimal_action = self._find_optimal_action()
        self.reward_variance = reward_variance
        self.reward_std = math.sqrt(self.reward_variance)
        self.stationary = stationary
        self.actions = actions

    def _find_optimal_action(self):
        return np.argmax(self.action_reward_means)

    def get_reward_for_action(self, a):
        reward = random.gauss(self.action_reward_means[a], self.reward_std)
        if not self.stationary:
            # Update all action reward means each time-step
            self.action_reward_means = self.action_reward_means + \
                        np.random.normal(0, 0.01, size=[self.actions, 1])
        self.optimal_action = self._find_optimal_action()
        return reward


def run_testbed(problems, actions_per_problem, explore_fraction,
                default_expected_value, timesteps):
    print(f"Training {len(problems)} agents on the testbed")
    agents = []
    for problem in tqdm(problems):
        agent = Agent(explore_fraction, actions_per_problem,
                      default_expected_value)
        for t in range(timesteps):
            agent.take_action(problem)
        agents.append(agent)
    return agents


def run_testbed_constant_alpha(problems, actions_per_problem, explore_fraction,
                default_expected_value, timesteps):
    print(f"Training {len(problems)} agents on the testbed")
    agents = []
    for problem in tqdm(problems):
        agent = AgentConstantAlpha(explore_fraction, actions_per_problem,
                      default_expected_value)
        for t in range(timesteps):
            agent.take_action(problem)
        agents.append(agent)
    return agents


if __name__ == '__main__':
    problem = BanditProblem(10, 0, 1, 1)
    agents = run_testbed([problem], 10, 0.01, 1.0, 10000)
    print("Done")
