from mesa import Model
from agent import PopAgent, Belief
from simpleScheduler import SimpleActivation
import random


class KnowledgeModel(Model):
    """A model with trust, controversy, and fallback around Method C."""

    def __init__(
        self,
        network,
        sharingMode,
        sharetime,
        delay,
        singleSource=False,
        samePartition=True,
        method_c_explainability=0.35,
        controversy_growth=0.08,
        controversy_decay=0.03,
        trust_sensitivity=0.5,
        fallback_threshold=0.55,
    ):
        super().__init__()

        self.mode = sharingMode
        self.G = network
        self.num_agents = self.G.number_of_nodes()
        self.schedule = SimpleActivation(self)

        self.delay = delay
        self.singleSource = singleSource
        self.samePartition = samePartition

        # Regulation / trust dynamics
        self.method_c_explainability = method_c_explainability
        self.controversy_growth = controversy_growth
        self.controversy_decay = controversy_decay
        self.trust_sensitivity = trust_sensitivity
        self.fallback_threshold = fallback_threshold

        self.controversy = 0.0
        self.trust = 1.0

        self.agentZero = None

        # Create agents
        for i in range(self.num_agents):
            neighbors = list(self.G.neighbors(i))
            a = PopAgent(
                unique_id=i,
                model=self,
                neighbors=neighbors,
                sharetime=sharetime,
            )

            if i == 0:
                a.belief = Belief.Method_A
                self.agentZero = a

            if i == 1 and self.delay == 0 and self.samePartition is None and self.singleSource:
                a.belief = Belief.Method_B

            if i == 2 and self.delay == 0 and self.samePartition is None and self.singleSource:
                a.belief = Belief.Method_C

            self.schedule.add(a)

        if self.delay == 0 and (self.samePartition is not None or not self.singleSource):
            self.addMethod_B()

        if self.delay == 0 and (self.samePartition is not None or not self.singleSource):
            self.addMethod_C()

    def addMethod_B(self):
        """Add Method_B belief to a selected agent."""
        if self.singleSource:
            a = self.agentZero
        elif self.samePartition is None:
            a = random.choice(self.schedule.agents)
        elif "partition" in self.G.graph and self.samePartition:
            if self.delay == 0:
                num = random.choice(list(self.G.graph["partition"][0])[1:])
            else:
                num = random.choice(list(self.G.graph["partition"][0]))
            a = self.schedule.agents[num]
        elif "partition" in self.G.graph and not self.samePartition:
            num = random.choice(list(self.G.graph["partition"][1]))
            a = self.schedule.agents[num]
        else:
            a = random.choice(self.schedule.agents)

        a.setBelief(Belief.Method_B)

    def addMethod_C(self):
        """Add Method_C belief to a selected agent."""
        if self.singleSource:
            a = self.agentZero
        elif self.samePartition is None:
            a = random.choice(self.schedule.agents)
        elif "partition" in self.G.graph and self.samePartition:
            if self.delay == 0:
                num = random.choice(list(self.G.graph["partition"][0])[1:])
            else:
                num = random.choice(list(self.G.graph["partition"][0]))
            a = self.schedule.agents[num]
        elif "partition" in self.G.graph and not self.samePartition:
            num = random.choice(list(self.G.graph["partition"][1]))
            a = self.schedule.agents[num]
        else:
            a = random.choice(self.schedule.agents)

        a.setBelief(Belief.Method_C)

    def update_regulation_state(self):
        """Update controversy and trust from the current belief distribution."""
        beliefs = [a.belief for a in self.schedule.agents]
        n = len(beliefs)

        frac_a = sum(1 for b in beliefs if b == Belief.Method_A) / n
        frac_b = sum(1 for b in beliefs if b == Belief.Method_B) / n
        frac_c = sum(1 for b in beliefs if b == Belief.Method_C) / n

        # High disagreement means controversy is more likely
        disagreement = 1.0 - max(frac_a, frac_b, frac_c)

        # Method C creates pressure when it is visible but hard to explain
        c_pressure = frac_c * (1.0 - self.method_c_explainability)

        self.controversy += self.controversy_growth * (c_pressure + disagreement)
        self.controversy -= self.controversy_decay
        self.controversy = max(0.0, min(1.0, self.controversy))

        self.trust = 1.0 - self.trust_sensitivity * self.controversy
        self.trust = max(0.0, min(1.0, self.trust))

    def step(self):
        """Advance the model by one step."""
        if self.delay > 0 and self.schedule.time == self.delay:
            self.addMethod_B()

        self.schedule.step()
        self.update_regulation_state()

    def logs(self):
        """Get agents' belief, interaction and pair history, respectively."""
        return self.schedule.logs()


if __name__ == "__main__":
    pass
