import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        #dictionary with states as keys, each state has four valid actions as nested dictionary.  each action contains
        # the q-value, for that state/action pair
        self.Q_table = {}
        self.valid_actions = self.env.valid_actions

        #Q-learning parameters -- pulled from env, because I intialized them there for the logger, csv file name.
        self.alpha = env.alpha
        self.gamma = env.gamma

        #logging variables
        self.reward = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward = 0

    #intialize all possible actions for a particular state in the Q-table.  Set = 0
    def init_Q_Values (self, state):
        possible_actions = {possible_action: 0 for possible_action in self.valid_actions }
        self.Q_table[state] = possible_actions

    #for the current state, iterates over all possible actions and returns the action which maximizes the Q value
    def ArgMAX_Q (self, state):
        for value in self.Q_table[state].values():
            if value != 0:  # this is will check each value, if any of them has a value then we've been to this state
                # before and can do argmax.  Otherwise we go below
               return max(self.Q_table[state].iteritems(), key= lambda x : x[1])[0]
        #otherwise, just pick a random action.  We have to force with random, because max doesn't randomly break ties,
        # it will choose the same option again and again for different unseen states.
        return random.choice(self.valid_actions)

    #for the current state, iterates over all possible actions and returns the largest Q value
    def MAX_Q (self, state):
        return max(self.Q_table[state].values())

    def epsilon_greedy(self, state, epsilon = 0.05):  #used to make random actions occasionally, increase exploration.
        if np.random.random() > epsilon:  #random is a float between [0,1).  If we set epsilon low, then we get restarts very occasionally.
            return self.ArgMAX_Q(state)
        else:
            return random.choice(self.valid_actions)


    def update(self, t):
        # Gather inputs   --- sense the environment
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        #this state will act as a key in the Q-table dictionary
        self.state = (('light', inputs['light'] ), ('next_waypoint', self.next_waypoint))

        #initialize the unseen states into the Q_table.
        if self.state not in self.Q_table:
            self.init_Q_Values(self.state)

        
        # TODO: Select action according to your policy
        # random_action = random.choice(self.env.valid_actions[1:])

        action = self.epsilon_greedy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward += reward
        #print self.reward

        # TODO: Learn policy based on state, action, reward

        #sense environment again,  in order to update Q_table.
        inputs_2 = self.env.sense(self)

        #deadline -= 1  #we have to force this because the deadline only updates with env.step(), it won't update from env.act().
        # Above code was for deadline in state, since removed because it increases state space too much.

        self.next_waypoint = self.planner.next_waypoint()
        state_2 = (('light', inputs_2['light'] ), ('next_waypoint', self.next_waypoint))

        #initialize the potentially unseen states into the Q_table.
        if state_2 not in self.Q_table:
            self.init_Q_Values(state_2)

        #Q-Learning, Simple, First Pass
        #self.Q_table[self.state][action] = (1-self.alpha) * self.Q_table[self.state][action] \
        #                                   + self.alpha * (reward + self.MAX_Q(state_2))

        # Q-learning
        self.Q_table[self.state][action] = (1-self.alpha) * self.Q_table[self.state][action] \
                                           + self.alpha * (reward + self.gamma * self.MAX_Q(state_2))

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run(alpha, gamma):
    """Run the agent for a finite number of trials."""


    # Set up environment and agent
    e = Environment(alpha =alpha, gamma = gamma)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials+    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    #close the log file
    e.file.close()


if __name__ == '__main__':
    #grid_values = np.arange(0,1.1,0.2) #use 1.1 in order to include 1, step is 0.1
    #for alpha in grid_values:
    #    for gamma in grid_values:
    #        run(alpha,gamma)

    run(0.8,0.2)
