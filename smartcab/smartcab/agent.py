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

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        #self.state = {"next waypoint" : self.next_waypoint, "Time Left" : deadline, "Inputs" : inputs}

        #this state will act as a key in the Q-table dictionary
        self.state = (('time_left', deadline), ('light', inputs['light'] ), ('next_waypoint', self.next_waypoint))

        
        # TODO: Select action according to your policy
        random_action = random.choice(self.env.valid_actions[1:])
        action = self.next_waypoint



        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        inputs_2 = self.env.sense(self)
        deadline_2 = self.env.get_deadline(self)
        state_2 = (('time_left', deadline), ('light', inputs['light'] ), ('next_waypoint', self.next_waypoint))

        valid_actions = self.env.valid_actions
        #dictionary with states as keys, each state has four valid actions as nested dictionary.  each action contains the q-value, for that state/action pair
        q_table = {}

        def Q_Value (state, action):
            if state in q_table:
                return q_table[state][action]
            else:
                possible_actions = {possible_action: 0 for possible_action in valid_actions }
                q_table[state] = possible_actions
            return q_table[state][action]


        def update_Q_table (state, action, prize, alpha, gamma):
            (1-alpha) * Q_Value(state, action) + alpha * (prize + gamma * MAX_Q(state, action))



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
