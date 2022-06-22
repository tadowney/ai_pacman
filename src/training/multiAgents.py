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
import numpy as np
import math
from util import Stack

from game import Agent

class Node:

    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def goalTest(self, gs, pos, flag):
        # Looking for food
        if(flag == 0):
            if(gs.hasFood(pos[0], pos[1])):
                return True
            return False
        # Looking for ghost
        if(flag == 1):
            gpos = gs.getGhostPositions()
            for gp in gpos:
                if(gp == pos):
                    return True
            return False
        

    def DLS(self, currentNode, stack, explored, layer, limit, found, flag):
        explored.append(currentNode)
        if(self.goalTest(currentNode.parent.state, currentNode.state.getPacmanPosition(), flag)):
            stack.push(currentNode)
            return stack, explored, True
        if(layer == limit):
            return stack, explored, False
        stack.push(currentNode)
        actions = currentNode.state.getLegalActions()
        for a in actions:
            newState = currentNode.state.generatePacmanSuccessor(a)
            newNode = Node(newState, currentNode, a, 1)
            if newNode in explored:
                continue
            stack, explored, found = self.DLS(newNode, stack, explored, layer+1, limit, found, flag)
            if(found):
                return stack, explored, True
        stack.pop()
        return stack, explored, False
    
    def IDS(self, sgs, limit, flag):
        found = False
        current_limit = 0
        while(not found and current_limit <= limit):
            current_limit = current_limit + 1
            startNode = Node(sgs, None, None, 0)
            startNode.parent = startNode
            stack = Stack()
            explored = []
            stack, explored, found = self.DLS(startNode, stack, explored, 1, current_limit, False, flag)

        actions = []
        while(not stack.isEmpty()):
            node = stack.pop()
            actions.append(node.action)

        if not actions:
            return actions, found
        
        actions.reverse()
        actions.pop(0)  # Removes start node from actions

        return actions, found


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        weights = np.loadtxt("weights.csv", delimiter=",")
        new_weights = np.array(weights)

        # Choose one of the best actions
        scores = []
        for action in legalMoves:
            [s,new_weights] = self.evaluationFunction(gameState, action, weights, new_weights)
            scores.append(s)
        
        bestScore = max(scores)
        allIndices = [index for index in range(len(scores))]
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #chosenIndex = random.choice(allIndices) # Pick randomly among the best


        "Add more of your code here if you want to"

        np.savetxt("weights.csv", new_weights, delimiter=",", fmt='%4.8f')

        return legalMoves[chosenIndex]

    def CalcGhostPos(self, cgs, actions):
        for a in actions:
            cgs = cgs.generatePacmanSuccessor(a)
        return cgs.getPacmanPosition()

    # Find all active and scared ghosts and then turn them into binary features
    def findAllGhosts(self, cgs):
        f1 = 0  # Active ghost one step away (Binary)
        f2 = 0  # Active ghost two steps away (Binary)
        f3 = 0  # Scared ghost one step away (Binary)
        f4 = 0  # Scared ghost two steps away (Binary)
        actions, found = self.IDS(cgs, 3, 1)
        if not found:
            return f1, f2, f3, f4
        ghosts = cgs.getGhostStates()
        ghostPos = self.CalcGhostPos(cgs, actions)
        foundGhostPosition = False
        for g in ghosts:
            if(ghostPos == g.configuration.pos):
                ghost = g
                foundGhostPosition = True
                break
        
        if not foundGhostPosition:
            return f1, f2, f3, f4

        if(ghost.scaredTimer > 0):  # If ghost is scared
            if(len(actions) <= 1):
                f3 = 1
            if(len(actions) == 2):
                f4 = 1
        if(ghost.scaredTimer == 0): # If ghost is active
            if(len(actions) <= 1):
                f1 = 1
            if(len(actions) == 2):
                f2 = 1

        return f1, f2, f3, f4


    # Active ghost one step away (Binary)
    def getFeatureOne(self, cgs):
        actions, found = self.IDS(cgs, 2, 1)
        if(found):
            if(len(actions) <= 1):
                return 1
        else:
            return 0

    # Active ghost two steps away (Binary)
    def getFeatureTwo(self, cgs):
        actions, found = self.IDS(cgs, 3, 1)
        if(found):
            if(len(actions) == 2):
                return 1
        else:
            return 0

    # Scared ghost one step away (Binary)
    def getFeatureThree(self, cgs):
        ghosts = cgs.getGhostStates()
        if not ghosts:
            return 0
        g = ghosts[0]
        if(g.scaredTimer > 0):
            actions, found = self.IDS(cgs, 2, 1)
            if(found):
                if(len(actions) <= 1):
                    return 1
            else:
                return 0
        return 0

    # Scared ghost two steps away (Binary)
    def getFeatureFour(self, cgs):
        ghosts = cgs.getGhostStates()
        if not ghosts:
            return 0
        g = ghosts[0]
        if(g.scaredTimer > 0):
            actions, found = self.IDS(cgs, 3, 1)
            if(found):
                if(len(actions) == 2):
                    return 1
            else:
                return 0
        return 0
        
    # Eating Food (Binary)
    def getFeatureFive(self, cgs, sgs):
        if(self.goalTest(cgs, sgs.getPacmanPosition(), 0)):
            return 1
        return 0

    # Distance to closest food
    def getFeatureSix(self, cgs):
        #actions, found = self.IDS(cgs, 3, 0)
        #if(found):
        #    return 1/len(actions)
        
        food = cgs.getFood()
        pacPos = cgs.getPacmanPosition()
        dist = []
        x_size = food.width
        y_size = food.height
        for x in range(0, x_size):
            for y in range(0, y_size):
                if(food[x][y] == True):
                    dist.append(manhattanDistance(pacPos, (x,y)))
        if not dist:
            return 0
        closestFood = min(dist)
        return 1/closestFood

    # Get instantaneous reward 
    def getReward(self, cgs, sgs):
        pacPos = sgs.getPacmanPosition()
        gpos = cgs.getGhostPositions()
        ghosts = cgs.getGhostStates()
        g = ghosts[0]
        for pos in gpos:
            if(pacPos == pos and g.scaredTimer == 0):
                return -250
            if(pacPos == pos and g.scaredTimer > 1):
                return 100
        if(cgs.hasFood(pacPos[0], pacPos[1])):
            if(cgs.getNumFood() <= 1):
                return 250
            return 1
        return -1
    

    def evaluationFunction(self, currentGameState, action, weights, new_weights):
       
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Get features for current state
        f1, f2, f3, f4 = self.findAllGhosts(successorGameState)
        f5 = self.getFeatureFive(currentGameState, successorGameState)
        f6 = self.getFeatureSix(successorGameState)
        features = np.array([f1, f2, f3, f4, f5, f6])
        Q_s_a = np.dot(weights, np.transpose(features))

        # Generate Q(s', a')
        Qs = []
        legalMoves = successorGameState.getLegalActions()
        for a in legalMoves:
            ngs = successorGameState.generatePacmanSuccessor(a)
            f1_next, f2_next, f3_next, f4_next = self.findAllGhosts(ngs) 
            f5_next = self.getFeatureFive(successorGameState, ngs)
            f6_next = self.getFeatureSix(ngs)
            features_next = np.array([f1_next, f2_next, f3_next, f4_next, f5_next, f6_next])
            Q_next = np.dot(weights, np.transpose(features_next))
            Qs.append(Q_next)
        if not Qs:
            Q_next = 0
        else:
            Q_next = max(Qs)

        r = self.getReward(currentGameState, successorGameState)

        alpha = 0.00001
        gamma = 0.9
        diff = (r + gamma*Q_next) - Q_s_a
        
        for w in range(0,6):
            new_weights[w] = new_weights[w] + alpha*diff*features[w]

        return [Q_s_a, new_weights]    # Q state calculated with old weights
    
    

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
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

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
