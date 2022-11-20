# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def pathGenerator(node):
    """
    Recursively generates the path of actions from a start state to a goal state.
    Yields actions of predecessor states first to preserve start to goal order.

    :param node: The current state
    :yield: The actions of the current states predecessors
    :yield: The action of the current state
    """

    if node["predecessor"]:
        yield from pathGenerator(node["predecessor"])
        yield node["action"]

def getActions(node):
    """
    Retrieves the list of actions from a start state to a goal state

    :param node: A goal state to construct the path from
    :return: A list of ordered actions
    """

    return [action for action in pathGenerator(node)]

def uninformedGraphSearch(problem, method):
    """
    Performs an uninformed search on a state space.
    Performs two types of uninformed search:
    - DFS: Pass util.Stack as the method
    - BFS: Pass util.Queue as the method

    :param problem: The problem or state space
    :param method: The type of uninformed search to perform
    """

    # If current state is goal state, need not take action
    if problem.isGoalState(start := problem.getStartState()):
        return []

    # Track visited states and fringe states
    visited = set()
    fringe = method()
    fringe.push({
        "state": start,
        "action": None,
        "predecessor": None
    })

    # While states within the fringe, explore
    while not fringe.isEmpty():

        # Get current node
        current = fringe.pop()

        # If goal state, return actions to reach goal state
        if problem.isGoalState(current["state"]):
            return getActions(current)

        # Prepare successors for exploration
        if current["state"] not in visited:
            visited.add(current["state"])
            for state, action, cost in problem.getSuccessors(current["state"]):
                if state not in visited:
                    fringe.push({
                        "state": state,
                        "action": action,
                        "predecessor": current
                    })

    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    return uninformedGraphSearch(problem, util.Stack)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    return uninformedGraphSearch(problem, util.Queue)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # UCS is a form of A*, where h(n) is the nullHeuristic
    return aStarSearch(problem, nullHeuristic)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # If current state is goal state, need not take action
    if problem.isGoalState(start := problem.getStartState()):
        return []

    # Track visited states and fringe states
    visited = set()
    fringe = util.PriorityQueue()
    fringe.push({
        "state": start,
        "action": None,
        "cost": 0,
        "predecessor": None
    }, 0)

    # While states within the fringe, explore
    while not fringe.isEmpty():

        # Get current node
        current = fringe.pop()

        # If goal state, return actions to reach goal state
        if problem.isGoalState(current["state"]):
            return getActions(current)

        # Prepare successors for exploration
        if current["state"] not in visited:
            visited.add(current["state"])
            for state, action, cost in problem.getSuccessors(current["state"]):
                if state not in visited:
                    costSoFar = cost + current["cost"]
                    estimatedCost = costSoFar + heuristic(state, problem)
                    fringe.push({
                        "state": state,
                        "predecessor": current,
                        "action": action,
                        "cost": costSoFar
                    }, estimatedCost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
