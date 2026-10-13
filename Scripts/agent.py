import pandas as pd
import random
from mesa import Agent
from belief import Belief, Mode

# Load thresholds from CSV
csv_file = "Highest_PR_Method_80-3P.csv"
df = pd.read_csv(csv_file)


def get_threshold(df, choice):
    """
    Return the threshold for a given choice as a numeric value.
    Raises a clear error if the value is missing or not numeric.
    """
    value = df.loc[df["Choice"] == choice, "Time"]

    if value.empty:
        raise ValueError(f"Missing threshold for choice '{choice}' in {csv_file}")

    threshold = pd.to_numeric(value.iloc[0], errors="coerce")

    if pd.isna(threshold):
        raise ValueError(
            f"Threshold for choice '{choice}' in {csv_file} is not numeric: {value.iloc[0]!r}"
        )

    return float(threshold)


NEUTRAL_TO_METHOD_A_THRESHOLD = get_threshold(df, "A")
METHOD_A_TO_METHOD_B_THRESHOLD = get_threshold(df, "B")
METHOD_B_TO_METHOD_C_THRESHOLD = get_threshold(df, "C")


class PopAgent(Agent):
    """An agent with a current belief and compact fallback dynamics."""

    def __init__(self, unique_id, model, neighbors, sharetime):
        super().__init__(unique_id, model)

        self.belief = Belief.Neutral
        self.neighbors = neighbors
        self.clock = 0
        self.beliefTime = 0
        self.shareTime = sharetime
        self.belief_transitions = []

    def tick(self):
        self.clock += 1
        self.beliefTime += 1

    def isSharing(self):
        return self.beliefTime <= self.shareTime

    def setBelief(self, belief):
        if self.belief != belief:
            transition = (self.clock, self.belief, belief)
            self.belief_transitions.append(transition)
            self.belief = belief
            self.beliefTime = 0

    def can_adopt_method_c(self):
        """
        Adoption of Method C depends on trust and controversy.
        Lower trust and higher controversy slow or block adoption.
        """
        adoption_prob = self.model.trust * (1.0 - 0.5 * self.model.controversy)
        adoption_prob = max(0.0, min(1.0, adoption_prob))
        return random.random() < adoption_prob

    def should_fallback_from_method_c(self):
        """
        If trust gets too low, Method C may revert to Method B.
        This is a simple proxy for revision / dilution / weak enforcement.
        """
        if self.model.trust >= self.model.fallback_threshold:
            return False

        fallback_prob = (
            (self.model.fallback_threshold - self.model.trust)
            + 0.3 * self.model.controversy
        )
        fallback_prob = max(0.0, min(1.0, fallback_prob))
        return random.random() < fallback_prob

    def update(self, other):
        """Update beliefs after interacting with another agent."""

        # Neutral -> A/B/C
        if self.belief == Belief.Neutral and other.belief == Belief.Method_A:
            self.setBelief(Belief.Method_A)

        if self.belief == Belief.Neutral and other.belief == Belief.Method_B:
            self.setBelief(Belief.Method_B)

        if self.belief == Belief.Neutral and other.belief == Belief.Method_C:
            if self.beliefTime > METHOD_B_TO_METHOD_C_THRESHOLD:
                if self.can_adopt_method_c():
                    self.setBelief(Belief.Method_C)
                else:
                    self.setBelief(Belief.Method_B)
            else:
                self.setBelief(Belief.Method_C)

        if (
            self.belief == Belief.Method_C
            and other.belief == Belief.Method_A
            and NEUTRAL_TO_METHOD_A_THRESHOLD < self.beliefTime <= METHOD_A_TO_METHOD_B_THRESHOLD
        ):
            self.setBelief(Belief.Method_A)

        if (
            self.belief == Belief.Method_B
            and other.belief == Belief.Method_A
            and NEUTRAL_TO_METHOD_A_THRESHOLD < self.beliefTime <= METHOD_A_TO_METHOD_B_THRESHOLD
        ):
            self.setBelief(Belief.Method_A)

        if (
            self.belief == Belief.Neutral
            and other.belief == Belief.Method_A
            and NEUTRAL_TO_METHOD_A_THRESHOLD < self.beliefTime <= METHOD_A_TO_METHOD_B_THRESHOLD
        ):
            self.setBelief(Belief.Method_A)

        # Reversion / movement toward B
        if (
            self.belief == Belief.Method_A
            and other.belief == Belief.Method_B
            and METHOD_A_TO_METHOD_B_THRESHOLD < self.beliefTime <= METHOD_B_TO_METHOD_C_THRESHOLD
        ):
            self.setBelief(Belief.Method_B)

        if (
            self.belief == Belief.Method_C
            and other.belief == Belief.Method_B
            and METHOD_A_TO_METHOD_B_THRESHOLD < self.beliefTime <= METHOD_B_TO_METHOD_C_THRESHOLD
        ):
            self.setBelief(Belief.Method_B)

        if (
            self.belief == Belief.Neutral
            and other.belief == Belief.Method_B
            and METHOD_A_TO_METHOD_B_THRESHOLD < self.beliefTime <= METHOD_B_TO_METHOD_C_THRESHOLD
        ):
            self.setBelief(Belief.Method_B)

        # Adoption of C with trust-sensitive fallback
        if (
            self.belief == Belief.Method_A
            and other.belief == Belief.Method_C
            and self.beliefTime > METHOD_B_TO_METHOD_C_THRESHOLD
        ):
            if self.can_adopt_method_c():
                self.setBelief(Belief.Method_C)
            else:
                self.setBelief(Belief.Method_B)

        if (
            self.belief == Belief.Method_B
            and other.belief == Belief.Method_C
            and self.beliefTime > METHOD_B_TO_METHOD_C_THRESHOLD
        ):
            if self.can_adopt_method_c():
                self.setBelief(Belief.Method_C)

        if (
            self.belief == Belief.Neutral
            and other.belief == Belief.Method_C
            and self.beliefTime > METHOD_B_TO_METHOD_C_THRESHOLD
        ):
            if self.can_adopt_method_c():
                self.setBelief(Belief.Method_C)
            else:
                self.setBelief(Belief.Method_B)

        if self.belief == Belief.Method_C and self.should_fallback_from_method_c():
            self.setBelief(Belief.Method_B)

    def step(self, interlocutor):
        self.update(interlocutor)


if __name__ == "__main__":
    pass
