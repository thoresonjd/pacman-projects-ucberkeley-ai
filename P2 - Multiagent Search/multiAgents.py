# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Set up food and capsules
        newFood = newFood.asList()
        newCapsules = successorGameState.getCapsules()

        # Prevent PacMan from stalling via direction lookahead; force PacMan to make progress
        currentDirection = currentGameState.getPacmanState().getDirection()
        successorDirection = successorGameState.getPacmanState().getDirection()

        # Rewards, penalties, capsule factor (a)
        foodReward, capsuleReward = 100, 200
        ghostReward, ghostPenalty = 100, -100
        idlePenalty = -50
        a = 2

        # Get all food, capsule, and ghost distances
        newFoodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood]
        newCapsuleDistances = [manhattanDistance(newPos, capsulePos) for capsulePos in newCapsules]
        newGhostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]

        # Set score; game over if no food remains
        score = successorGameState.getScore()
        if not newFoodDistances:
            return score

        # Find nearest food, capsule, and ghost
        nearestFoodDistance = min(newFoodDistances)
        nearestCapsuleDistance = min(newCapsuleDistances) if newCapsuleDistances else float('inf')
        nearestGhostDistance = min(newGhostDistances) if newGhostDistances else float('inf')
        scaredTimer = newScaredTimes[newGhostDistances.index(nearestGhostDistance)] if newGhostDistances else 0

        # Reward and/or penalize
        score += foodReward if not nearestFoodDistance else 1/nearestFoodDistance
        score += capsuleReward if not nearestCapsuleDistance else a/nearestCapsuleDistance
        score += ghostReward if scaredTimer > 0 else ghostPenalty if nearestGhostDistance <= 1 else 0
        score += idlePenalty if action == 'Stop' or currentDirection == Directions.REVERSE[successorDirection] else 0

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Adapted from Minimax pseudocode within lecture slides

        def isTerminalState(gameState, legalActions, depth):
            return gameState.isWin() or gameState.isLose() or not legalActions or depth >= self.depth

        def value(gameState, agent, depth):
            if isTerminalState(gameState, legalActions := gameState.getLegalActions(agent), depth):
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maxValue(gameState, agent, legalActions, depth)
            if agent > 0:
                return minValue(gameState, agent, legalActions, depth)

        def maxValue(gameState, agent, legalActions, depth):
            maximumValue = float('-inf')
            for legalAction in legalActions:
                successorState = gameState.generateSuccessor(agent, legalAction)
                maximumValue = max(value(successorState, agent + 1, depth), maximumValue)
            return maximumValue

        def minValue(gameState, agent, legalActions, depth):
            minimumValue = float('inf')
            for legalAction in legalActions:
                successorState = gameState.generateSuccessor(agent, legalAction)
                if agent == gameState.getNumAgents() - 1:
                    minimumValue = min(value(successorState, 0, depth + 1), minimumValue)
                else:
                    minimumValue = min(value(successorState, agent + 1, depth), minimumValue)
            return minimumValue

        pacmanAgent, firstGhostAgent, startingDepth = 0, 1, 0
        utility, action = float('-inf'), None
        for legalAction in gameState.getLegalActions(pacmanAgent):
            successorState = gameState.generateSuccessor(pacmanAgent, legalAction)
            if (newUtility := value(successorState, firstGhostAgent, startingDepth)) > utility:
                utility = newUtility
                action = legalAction
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Adapted from ABMinimax pseudocode within lecture slides

        def isTerminalState(gameState, legalActions, depth):
            return gameState.isWin() or gameState.isLose() or not legalActions or depth >= self.depth

        def value(gameState, agent, depth, alpha, beta):
            if isTerminalState(gameState, legalActions := gameState.getLegalActions(agent), depth):
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maxValue(gameState, agent, legalActions, depth, alpha, beta)
            if agent > 0:
                return minValue(gameState, agent, legalActions, depth, alpha, beta)

        def maxValue(gameState, agent, legalActions, depth, alpha, beta):
            maximumValue = float('-inf')
            for legalAction in legalActions:
                successorState = gameState.generateSuccessor(agent, legalAction)
                maximumValue = max(value(successorState, agent + 1, depth, alpha, beta), maximumValue)
                if maximumValue > beta:
                    return maximumValue
                alpha = max(alpha, maximumValue)
            return maximumValue

        def minValue(gameState, agent, legalActions, depth, alpha, beta):
            minimumValue = float('inf')
            for legalAction in legalActions:
                successorState = gameState.generateSuccessor(agent, legalAction)
                if agent == gameState.getNumAgents() - 1:
                    minimumValue = min(value(successorState, 0, depth + 1, alpha, beta), minimumValue)
                else:
                    minimumValue = min(value(successorState, agent + 1, depth, alpha, beta), minimumValue)
                if minimumValue < alpha:
                    return minimumValue
                beta = min(beta, minimumValue)
            return minimumValue

        pacmanAgent, firstGhostAgent, startingDepth = 0, 1, 0
        alpha, beta = float('-inf'), float('inf')
        action = None
        for legalAction in gameState.getLegalActions(pacmanAgent):
            successorState = gameState.generateSuccessor(pacmanAgent, legalAction)
            if (newUtility := value(successorState, firstGhostAgent, startingDepth, alpha, beta)) > alpha:
                alpha = newUtility
                action = legalAction
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Adapted from Expectimax pseudocode within lecture slides

        def isTerminalState(gameState, legalActions, depth):
            return gameState.isWin() or gameState.isLose() or not legalActions or depth >= self.depth

        def value(gameState, agent, depth):
            if isTerminalState(gameState, legalActions := gameState.getLegalActions(agent), depth):
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maxValue(gameState, agent, legalActions, depth)
            if agent > 0:
                return expValue(gameState, agent, legalActions, depth)

        def maxValue(gameState, agent, legalActions, depth):
            maximumValue = float('-inf')
            for legalAction in legalActions:
                successorState = gameState.generateSuccessor(agent, legalAction)
                maximumValue = max(value(successorState, agent + 1, depth), maximumValue)
            return maximumValue

        def expValue(gameState, agent, legalActions, depth):
            expectedValue = 0
            for legalAction in legalActions:
                successorState = gameState.generateSuccessor(agent, legalAction)
                probability = 1 / len(legalActions)
                if agent == gameState.getNumAgents() - 1:
                    expectedValue += probability * value(successorState, 0, depth + 1)
                else:
                    expectedValue += probability * value(successorState, agent + 1, depth)
            return expectedValue

        pacmanAgent, firstGhostAgent, startingDepth = 0, 1, 0
        utility, action = float('-inf'), None
        for legalAction in gameState.getLegalActions(pacmanAgent):
            successorState = gameState.generateSuccessor(pacmanAgent, legalAction)
            if (newUtility := value(successorState, firstGhostAgent, startingDepth)) > utility:
                utility = newUtility
                action = legalAction
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    This evaluation function is similar to the evaluation function used in the
    ReflexAgent. Since no action is provided, and only the current game state
    is known, everything involving a successor state now checks current state
    instead. Furthermore, since the current state cannot be compared to a
    successor state, and since no action is given, PacMan cannot be prevented
    from idling or stalling when the action is STOP or when the successor
    direction is the opposite of the current direction. Therefore, this is not
    possible here as it was with the ReflexAgent.
    """
    "*** YOUR CODE HERE ***"

    # Game state extraction
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newCapsules = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Rewards, penalties, capsule factor (a)
    foodReward, capsuleReward = 100, 200
    ghostReward, ghostPenalty = 100, -100
    a = 2

    # Get all food, capsule, and ghost distances
    newFoodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood]
    newCapsuleDistances = [manhattanDistance(newPos, capsulePos) for capsulePos in newCapsules]
    newGhostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]

    # Set score; game over if no food remains
    score = currentGameState.getScore()
    if not newFoodDistances:
        return score

    # Find nearest food, capsule, and ghost
    nearestFoodDistance = min(newFoodDistances)
    nearestCapsuleDistance = min(newCapsuleDistances) if newCapsuleDistances else float('inf')
    nearestGhostDistance = min(newGhostDistances) if newGhostDistances else float('inf')
    scaredTimer = newScaredTimes[newGhostDistances.index(nearestGhostDistance)] if newGhostDistances else 0

    # Reward and/or penalize
    score += foodReward if not nearestFoodDistance else 1 / nearestFoodDistance
    score += capsuleReward if not nearestCapsuleDistance else a/nearestCapsuleDistance
    score += ghostReward if scaredTimer > 0 else ghostPenalty if nearestGhostDistance <= 1 else 0

    return score

# Abbreviation
better = betterEvaluationFunction
