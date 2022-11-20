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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        # Perform iterations
        for i in range(self.iterations):

            # Keep offline copy of values, update after each iteration
            offlineValues = self.values.copy()

            # For each state, run the bellman equation
            for state in states:
                maxValue, bestAction = self.bellmanEquation(state)
                if maxValue > float('-inf'):
                    offlineValues[state] = maxValue

            # Update values with offline copy
            self.values = offlineValues.copy()

    def bellmanEquation(self, state):
        maxValue, bestAction = float('-inf'), None
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            if (qSummation := self.qValue(state, action)) > maxValue:
                maxValue = qSummation
                bestAction = action
        return maxValue, bestAction

    def qValue(self, state, action):
        qValue = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, probability in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            nextStateValue = self.getValue(nextState)
            qValue += probability * (reward + self.discount * nextStateValue)
        return qValue

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

        qValue = self.qValue(state, action)
        return qValue


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
        maxValue, bestAction = self.bellmanEquation(state)
        return bestAction

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

        # Execute Bellman equation on one state per iteration
        for i in range(self.iterations):
            state = states[i % len(states)]
            maxValue, bestAction = self.bellmanEquation(state)
            if maxValue > float('-inf'):
                self.values[state] = maxValue

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

        states = self.mdp.getStates()
        predecessors = self.computePredecessorsForAllStates(states)
        pq = util.PriorityQueue()
        self.initializeStatePrioritizations(pq, states)
        self.prioritySweep(pq, predecessors)

    def computePredecessorsForAllStates(self, states):
        predecessors = {}
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, probability in transitions:
                    if probability > 0:
                        if nextState not in predecessors:
                            predecessors[nextState] = set()
                        predecessors[nextState].add(state)
        return predecessors

    def computeMaxQValue(self, state):
        actions = self.mdp.getPossibleActions(state)
        maxQValue = float('-inf')
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            maxQValue = max(qValue, maxQValue)
        return maxQValue

    def initializeStatePrioritizations(self, pq, states):
        for state in states:
            if not self.mdp.isTerminal(state):
                currentValue, qValue = self.values[state], self.computeMaxQValue(state)
                diff = abs(currentValue - qValue)
                pq.push(state, -diff)

    def prioritySweep(self, pq, predecessors):
        for iteration in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = self.computeMaxQValue(state)
            for predecessor in predecessors[state]:
                predecessorCurrentValue = self.values[predecessor]
                predecessorQValue = self.computeMaxQValue(predecessor)
                diff = abs(predecessorCurrentValue - predecessorQValue)
                if diff > self.theta:
                    pq.update(predecessor, -diff)