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
import math

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
        # Note: As features, try the reciprocal of important values (such as distance to food) rather than just the values themselves.
        nearest_food = math.inf
        for food in newFood.asList():
            if manhattanDistance(newPos, food) < nearest_food:
                nearest_food = manhattanDistance(newPos, food)


        ghosts = successorGameState.getGhostPositions()
        for g in ghosts:
            if manhattanDistance(g, newPos) <= 1:
                return -math.inf


        return successorGameState.getScore() + (1/nearest_food)*2

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
        return self.maxnode(gameState, 0, 0)[0]

    def minimax(self, game_state, depth, agent_type):
        if depth >= self.depth * game_state.getNumAgents() or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)
        if agent_type == 0:
            return self.maxnode(game_state, depth, agent_type)[1]
        else:
            return self.minnode(game_state, depth, agent_type)[1]

    def maxnode(self, game_state, depth, agent_type):
        max_choice = ('None', -math.inf)

        for action in game_state.getLegalActions(agent_type):
            candidate = (action, self.minimax(game_state.generateSuccessor(agent_type, action), depth+1, (depth + 1) % game_state.getNumAgents()))

            if candidate[1] >= max_choice[1]:
                max_choice = candidate

        return max_choice

    def minnode(self, game_state, depth, agent_type):
        min_choice = ('None', math.inf)
        for action in game_state.getLegalActions(agent_type):
            candidate = (action, self.minimax(game_state.generateSuccessor(agent_type, action), depth+1, (depth + 1) % game_state.getNumAgents()))

            if candidate[1] <= min_choice[1]:
                min_choice = candidate

        return min_choice


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxnode(gameState, 0 , 0, -math.inf, math.inf)[0]

    def alpha_beta(self, gameState, depth, agent_type, alpha, beta):
        if depth >= self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agent_type == 0:  # If agent is maximizer
            return self.maxnode(gameState, depth, agent_type, alpha, beta)[1]

        else:  # If agent is minimizer
            return self.minnode(gameState, depth, agent_type, alpha, beta)[1]

    def maxnode(self, gameState, depth, agent_type, alpha, beta):
        best_action = None
        best_val = -math.inf

        for action in gameState.getLegalActions(agent_type):
            next_state = gameState.generateSuccessor(agent_type, action)
            next_agent = (depth + 1) % gameState.getNumAgents()
            candidate = self.alpha_beta(next_state, depth+1, next_agent, alpha, beta)

            if candidate >= best_val:
                best_val = candidate
                best_action = action

            # Prune
            if best_val > beta: return (best_action, best_val)
            else: alpha = max(alpha, best_val)

        return (best_action, best_val)

    def minnode(self, gameState, depth, agent_type, alpha, beta):
        best_action = None
        best_val = math.inf
        for action in gameState.getLegalActions(agent_type):
            next_state = gameState.generateSuccessor(agent_type, action)
            next_agent = (depth + 1) % gameState.getNumAgents()
            candidate = self.alpha_beta(next_state, depth+1, next_agent, alpha, beta)

            if candidate <= best_val:
                best_val = candidate
                best_action = action

            # Pruning
            if best_val < alpha: return (best_action, best_val)
            else: beta = min(beta, best_val)

        return (best_action, best_val)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, self.depth * gameState.getNumAgents(), 0, 'haha')[0]

    def expectimax(self, gameState, depth, agent_type, action):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return (action, self.evaluationFunction(gameState))
        if agent_type == 0:  # Maximizer
            return self.maxnode(gameState, depth, agent_type, action)
        else:  # Expectation node
            return self.expnode(gameState, depth, agent_type, action)

    def expnode(self, gameState, depth, agent_type, action):
        candidates = gameState.getLegalActions(agent_type)
        e_val = 0
        for a in candidates:
            next_agent = (agent_type + 1) % gameState.getNumAgents()
            next_state = gameState.generateSuccessor(agent_type, a)
            best_move = self.expectimax(next_state, depth-1, next_agent, action)
            e_val += best_move[1] * (1/len(candidates))

        return (action, e_val)

    def maxnode(self, gameState, depth, agent_type, action):
        best_move = ("None", -math.inf)
        for a in gameState.getLegalActions(agent_type):
            next_agent = (agent_type + 1) % gameState.getNumAgents()
            next_state = gameState.generateSuccessor(agent_type, a)
            if depth != self.depth * gameState.getNumAgents():
                candidate = action
            else:
                candidate = a
            cand_val = self.expectimax(next_state, depth-1, next_agent, candidate)
            if cand_val[1] >= best_move[1]:
                best_move = cand_val
        return best_move



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
