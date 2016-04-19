import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        #dictionary with states as keys, each state has four valid actions as nested dictionary.  each action contains the q-value, for that state/action pair
        self.Q_table = {}
        self.valid_actions = self.env.valid_actions


        #Q learning parameters
        self.gamma = .5
        self.alpha = .05

        #csv file title name
        self.trial_parameters = "Gamma={}_Alpha={}".format(self.gamma,self.alpha)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    #intialize all possible actions for a particular state in the Q-table.  Set = 0
    def init_Q_Values (self, state):
        possible_actions = {possible_action: 0 for possible_action in self.valid_actions }
        self.Q_table[state] = possible_actions

    #for the current state, iterates over all possible actions and returns the action which maximizes the Q value
    def ArgMAX_Q (self, state):
        #if the sum of the values is greater than 0, then there is an entry, so choose the max
        if sum(self.Q_table[state].values()) != 0:
               return max(self.Q_table[state].iteritems(), key= lambda x : x[1])[0]
        else: #otherwise, just pick a random action.  We have to force this because max doesn't randomly break ties, it will choose the same option again and again for different unseen states.
            return random.choice(self.valid_actions)

    #for the current state, iterates over all possible actions and returns the largest Q value
    def MAX_Q (self, state):
        return max(self.Q_table[state].values())



    def update(self, t):
        # Gather inputs   --- sense the environment
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        #this state will act as a key in the Q-table dictionary
        self.state = (('light', inputs['light'] ), ('next_waypoint', self.next_waypoint))

        #initialize the state into the Q_table will do nothing if it's already there.
        if self.state not in self.Q_table:
            self.init_Q_Values(self.state)

        
        # TODO: Select action according to your policy
        # random_action = random.choice(self.env.valid_actions[1:])

        action = self.ArgMAX_Q(self.state)


        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        #sense environment again,  in order to update Q_table.
        inputs_2 = self.env.sense(self)
        #deadline -= 1  #we have to force this because the deadline only updates with env.step(), it won't update from env.act().
        self.next_waypoint = self.planner.next_waypoint()
        state_2 = (('light', inputs_2['light'] ), ('next_waypoint', self.next_waypoint))

        #initialize the new state into the Q_table will do nothing if it's already there.
        if state_2 not in self.Q_table:
            self.init_Q_Values(state_2)


        # Q-learning
        self.Q_table[self.state][action] = (1-self.alpha) * self.Q_table[self.state][action] \
                                           + self.alpha * (reward + self.gamma * self.MAX_Q(state_2))




        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""


    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials+    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)

    #initiliaze file for CSV logging
    with open('smartCabLog.txt', 'ab') as log:
        log.write("\n-----------------------------------")
        log.write("\nAlpha is set to {}".format(a.alpha))
        log.write("\nGamma is set to {}".format(a.gamma))

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
