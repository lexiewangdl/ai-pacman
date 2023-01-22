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
    # Initialize fringe
    fringe = util.Stack()
    # Add start state and path to fringe
    fringe.push((problem.getStartState(), []))
    checked = []  # List to store checked nodes

    if problem.isGoalState(problem.getStartState()):
        print("Start state is the goal state.")
        return []

    while not fringe.isEmpty():
        curr, path = fringe.pop() # curr is a tuple with (x,y) location of agent
                                  # path is a list: path from start to curr
        if curr not in checked:
            checked.append(curr)
            # TERMINATE: Reached goal state, return path to get to here
            if problem.isGoalState(curr):
                return path
            # Have not reached goal state yet, continue searching
            successors = problem.getSuccessors(curr)
            for item in successors:
                # item[0] is (x,y) location of successor
                # item[1] is action to get to successor, e.g. 'South'
                updatedPath = path.copy() # Create copy of original path, avoid changing path variable
                updatedPath.append(item[1]) # Add step taken to the list of path
                fringe.push((item[0], updatedPath))

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    # Initialize fringe
    fringe = util.Queue()  # Fringe in BFS is FIFO queue
    checked = []

    fringe.push((start_state, []))

    while not fringe.isEmpty():
        curr, path = fringe.pop()

        if curr not in checked:
            checked.append(curr)
            if problem.isGoalState(curr):
                return path
            successors = problem.getSuccessors(curr)
            for item in successors:
                # item[0] is (x,y) location of successor
                # item[1] is action to get to successor, e.g. 'South'
                updatedPath = path.copy()  # Create copy of original path, avoid changing path variable
                updatedPath.append(item[1])  # Add step taken to the list of path
                fringe.push((item[0], updatedPath))

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    # In UCS, fringe is a priority queue
    fringe = util.PriorityQueue()
    fringe.push((start_state, [], 0), 0)

    checked = []

    while not fringe.isEmpty():
        curr, path, cost = fringe.pop()

        if curr not in checked:
            checked.append(curr)
            if problem.isGoalState(curr):
                return path
            successors = problem.getSuccessors(curr)
            for item in successors:
                # item[0] is (x,y) location of successor
                # item[1] is action to get to successor, e.g. 'South'
                # item[2] is cost
                updatedPath = path.copy()  # Create copy of original path, avoid changing path variable
                updatedPath.append(item[1])  # Add step taken to the list of path
                p = problem.getCostOfActions(updatedPath)  # Returns the cost of a particular sequence of actions.
                fringe.push((item[0], updatedPath, p), p)  # push(self, item, priority)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    # In A* Search, we both look at path cost and goal proximity
    # For each move, look at path cost in priority queue
    # Goal proximity is estimated with heuristic
    fringe = util.PriorityQueue()
    fringe.push((start_state, [], 0), 0)

    checked = []

    while not fringe.isEmpty():
        curr, path, past_cost = fringe.pop()

        if curr not in checked:
            checked.append(curr)

            if problem.isGoalState(curr):
                return path

            # Look at next steps
            successors = problem.getSuccessors(curr)
            for item in successors:
                # item[0] is (x,y) location of successor
                # item[1] is action to get to successor, e.g. 'South'
                # item[2] is cost
                updatedPath = path.copy()  # Create copy of original path, avoid changing path variable
                updatedPath.append(item[1])  # Add step taken to the list of path
                # f(n) = g(n) + h(n)
                # getCostOfActions Returns the cost of a particular sequence of actions, g(n)
                # heuristic returns the forward cost/goal proximity, h(n)
                p = problem.getCostOfActions(updatedPath) + heuristic(item[0], problem)
                fringe.push((item[0], updatedPath, p), p)  # push(self, item, priority)

    return []
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
