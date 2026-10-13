from mesa.time import BaseScheduler
from simpleLogger import SimpleLogger
import random

class SimpleActivation(BaseScheduler):
    """ A simple scheduler which randomly picks only one pair of agents,
    activating their step function."""

    def __init__(self, model):
        super().__init__(model)
        self._agents = []  # Initialize as a list
        # create logger
        self.logger = SimpleLogger(model)

    def add(self, agent):
        """Add an Agent object to the schedule and logger."""
        self._agents.append(agent)
        self.logger.add(agent)

    def logs(self):
        """Get agents' belief and interaction history, respectively."""
        return self.logger.logs()

    def choose(self):
        """Chooses pair of neighboring agents for interaction."""
        # pick agent A
        agentA = random.choice(self._agents)

        # pick agent B
        agentB_id = random.choice(agentA.neighbors)
        agentB = self._agents[agentB_id]

        return agentA, agentB

    def step(self):
        """Increments the timer for all agents, then lets one pair of agents interact."""
        # Increment timers
        for agent in self._agents:
            agent.tick()

        # choose agent pair
        agentA, agentB = self.choose()

        # interact
        agentA.step(agentB)
        agentB.step(agentA)

        # log results
        self.logger.log(agentA, agentB)

        # increment counters
        self.steps += 1
        self.time += 1

    @property
    def agents(self):
        """Returns the list of agents."""
        return self._agents
