# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.policy = util.Counter()

        self.runValueIteration()


    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            new_values = self.values.copy()
            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)
                if len(actions) == 0:
                    continue
                q_vals = []
                for action in actions:
                    q_val = self.getQValue(state, action)
                    q_vals.append(q_val)
                new_values[state] = max(q_vals)
            self.values = new_values

        # Compute the optimal policy
        for state in self.mdp.getStates():
            actions = self.mdp.getPossibleActions(state)
            if len(actions) == 0:
                continue
            max_q_val = float("-inf")
            best_action = None
            for action in actions:
                q_val = self.getQValue(state, action)
                if q_val > max_q_val:
                    max_q_val = q_val
                    best_action = action
            self.policy[state] = best_action
        return


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, prob in transitions:
            reward = self.mdp.getReward(state, action, next_state)
            value = self.getValue(next_state)
            discount = self.discount
            q_val += prob * (reward + discount * value)
        return q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        q_vals = util.Counter()
        for possibleAction in actions:
            q_vals[possibleAction] = self.computeQValueFromValues(state, possibleAction)
        decision = q_vals.argMax()
        return decision


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        state_idx = 0
        for i in range(0, self.iterations):
            if state_idx == len(states):
                state_idx = 0
            state = states[state_idx]
            state_idx += 1

            if self.mdp.isTerminal(state):
                continue
            action = self.computeActionFromValues(state)
            q_val = self.computeQValueFromValues(state, action)
            self.values[state] = q_val


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        queue = util.PriorityQueue()
        pred = util.Counter()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                pred[state] = set()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):  # Next step if state is terminal state
                continue
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for next_state, prob in transitions:
                    if (prob != 0) and not self.mdp.isTerminal(next_state):
                        pred[next_state].add(state)

            # calculate priority and push into queue
            val = self.values[state]
            best_action = self.computeActionFromValues(state)
            q_val = self.computeQValueFromValues(state, best_action)
            diff = abs(val - q_val)
            queue.push(state, -diff)

        for it in range(0, self.iterations):
            if queue.isEmpty():
                return
            s = queue.pop()
            # calculate Q-value for updating s
            best_action = self.computeActionFromValues(s)
            self.values[s] = self.computeQValueFromValues(s, best_action)
            for p in pred[s]:
                val = self.values[p]
                best_action = self.computeActionFromValues(p)
                q_val = self.computeQValueFromValues(p, best_action)
                diff = abs(val - q_val)
                if diff > self.theta:
                    queue.update(p, -diff)

