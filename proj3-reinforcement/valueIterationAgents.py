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
            q_val += prob * (reward + self.discount * value)
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
        return self.policy[state] if state in self.policy else None


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
        for i in range(self.iterations):
            for s in states:
                Q_vals = util.Counter()
                if self.mdp.isTerminal(s):
                    continue
                if len(self.mdp.getPossibleActions(s)) == 0:
                    continue
                for action in self.mdp.getPossibleActions(s):
                    Q_vals[action] = self.getQValue(s, action)
                self.values[s] = Q_vals[Q_vals.argMax()]
        return


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
        # TODO: MAKE CHANGES
        # Initialize an empty priority queue
        self.queue = util.PriorityQueue()
        self.predecessors = util.Counter()

        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                self.predecessors[s] = set()

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue

            # compute predecessors for state s
            possibleActions = self.mdp.getPossibleActions(s)
            for action in possibleActions:
                nextTransitions = self.mdp.getTransitionStatesAndProbs(s, action)
                for nextState, prob in nextTransitions:
                    if prob != 0 and not self.mdp.isTerminal(nextState):
                        self.predecessors[nextState].add(s)

            # calculate priority and push into queue
            currentValue = self.values[s]
            bestAction = self.computeActionFromValues(s)
            highestQValue = self.computeQValueFromValues(s, bestAction)
            diff = abs(currentValue - highestQValue)
            self.queue.push(s, -diff)

        for iter in range(0, self.iterations):
            if self.queue.isEmpty():
                # terminate
                return

            s = self.queue.pop()

            # calculate Q-value for updating s
            bestAction = self.computeActionFromValues(s)
            self.values[s] = self.computeQValueFromValues(s, bestAction)

            for p in self.predecessors[s]:
                currentValue = self.values[p]
                bestAction = self.computeActionFromValues(p)
                highestQValue = self.computeQValueFromValues(p, bestAction)
                diff = abs(currentValue - highestQValue)
                if diff > self.theta:
                    self.queue.update(p, -diff)

