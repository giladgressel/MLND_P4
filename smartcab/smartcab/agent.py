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


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required


    def Q_Value (self, state, action):
        if state in self.Q_table:
            return self.Q_table[state][action]
        else:
            possible_actions = {possible_action: 0 for possible_action in self.valid_actions }
            self.Q_table[state] = possible_actions
        return self.Q_table[state][action]

    #for the current state, iterates over all possible actions and returns the action which maximizes the Q value
    def ArgMAX_Q (self, state):
        
        return max(self.Q_table[state].iteritems(), key= lambda x : x[1])


    #for the current state, iterates over all possible actions and returns the largest Q value
    def MAX_Q (self, state):
        return max(self.Q_table[state].values())



    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        #this state will act as a key in the Q-table dictionary
        self.state = (('time_left', deadline), ('light', inputs['light'] ), ('next_waypoint', self.next_waypoint))

        
        # TODO: Select action according to your policy
        # random_action = random.choice(self.env.valid_actions[1:])

        action = self.ArgMAX_Q(self.state)



        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        #sense environment again,  in order to update Q_table.
        inputs_2 = self.env.sense(self)
        deadline -= 1  #we have to force this because the deadline only updates with env.step(), it won't update from env.act().
        self.next_waypoint = self.planner.next_waypoint()
        state_2 = (('time_left', deadline), ('light', inputs['light'] ), ('next_waypoint', self.next_waypoint))





        def update_Q_table (state_1, state_2, action_1, action_2, prize, alpha, gamma):
            new_q = (1-alpha) * Q_Value(state_1, action_1) + alpha * (prize + gamma * MAX_Q(state_2, action_2))



        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=3.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
