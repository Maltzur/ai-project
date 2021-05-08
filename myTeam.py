# myTeam.py
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

from __future__ import division

import json
import os
import random
import time

# import memcache

import util
from captureAgents import CaptureAgent
from game import Directions, Actions
from util import manhattanDistance, Queue

# cache = memcache.Client(['127.0.0.1:11211'], debug=1)


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='OffensiveAgent', second='DefensiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


# def read_weights():
#     weights = cache.get('weights')
#     if weights:
#         return weights

#     while True:
#         try:
#             with open('weights.json', 'r') as f:
#                 return json.load(f)
#         except OSError:
#             time.sleep(0.25)


# def write_weights(weights):
#     cache.set('weights', weights)

#     while True:
#         try:
#             with open('weights.json', 'w') as f:
#                 json.dump(weights, f)
#                 return
#         except OSError:
#             time.sleep(0.25)


##########
# Agents #
##########

class BaseAgent(CaptureAgent, object):

    # @property
    # def weights(self):
    #     weights = cache.get('weights')
    #     return weights

    # @weights.setter
    # def weights(self, value):
    #     cache.set('weights', value)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        CaptureAgent.registerInitialState(self, gameState)

        # Width of the board
        self.width = gameState.data.layout.width

        # Height of the board
        self.height = gameState.data.layout.height

        # Middle of the board (my side)
        self.mid = gameState.data.layout.width // 2 - int(self.red)

        # Get the ally index
        self.ally = next(iter(filter(lambda x: x != self.index, self.getTeam(gameState))))

        # Get the indices of the baddies
        self.opponents = self.getOpponents(gameState)

        # Get the legal positions for agents to be in
        self.legalPositions = set(gameState.getWalls().asList(False))

        # Get food placement at start
        self.initialFood = self.getFood(gameState).asList()

        # Find the longest path length
        self.maxPathLength = max(self.distancer._distances.values())

        # successor = gameState.generateSuccessor(self.index, action)
        # myPos = successor.getAgentPosition(self.index)
        # mid = self.width / 2 - int(gameState.isOnRedTeam(self.index))
        # distanceFromStart = min(
        #     self.distancer.getDistance(myPos, (mid, i))
        #     for i in range(self.height)
        #     if (mid, i) in self.legalPositions
        # )

        # Find the longest path length to my side
        self.maxSafeLength = 0
        for pos in self.legalPositions:
            if self.red ^ gameState.isRed(pos):
                for y in range(self.height):
                    midPos = (self.mid, y)
                    if midPos in self.legalPositions:
                        self.maxSafeLength = max(self.getMazeDistance(pos, midPos), self.maxSafeLength)

        # Find the number of crossings on the board
        self.numCrossings = 0
        walls = gameState.getWalls()
        for pos in self.legalPositions:
            if len(Actions.getLegalNeighbors(pos, walls)) >= 4:
                self.numCrossings += 1

        self.deadends = []
        walls = gameState.getWalls()
        for pos in self.legalPositions:
            if self.notOnMySide(gameState, pos) and len(Actions.getLegalNeighbors(pos, walls)) == 2:
                self.deadends.append(pos)

        self.deadendPaths = []
        for deadend in self.deadends:
            visited, queue = set(), Queue()

            visited.add(deadend)
            queue.push((deadend, []))

            while not queue.isEmpty():
                pos, path = queue.pop()

                neighbors = Actions.getLegalNeighbors(pos, walls)

                if len(neighbors) >= 4:
                    # path = list(set(path + neighbors))
                    break

                for n_pos in neighbors:
                    if n_pos not in visited:
                        visited.add(n_pos)
                        queue.push((n_pos, path + [n_pos]))

            self.deadendPaths.append(path)

        self.maxEntrapmentLength = max(map(len, self.deadendPaths))

        # Q Learning
        self.epsilon = 0.0
        self.alpha = 0.005
        self.discountFactor = 0.95

        # if 'entrapment' not in self.weights:
        #     weights = self.weights
        #     weights['entrapment'] = 0
        #     self.weights = weights
        # self.weights = {
        #     'powerpill': 0,
        #     'food': 0.1,
        #     'ghost': 0,
        #     'scared': 0,
        #     'reverse-action': 0,
        #     'risk': 0,
        # }
        # self.weights = read_weights()
        self.weights = {
            'powerpill': 0.2,
            'food': 0.303,
            'ghost': -0.3,
            'scared': 0.3,
            'reverse-action': -0.1,
            'risk': 0.45,
            'entrapment': -0.016,
        }

    def notOnMySide(self, gameState, pos):
        return self.red ^ gameState.isRed(pos)

    def expectimax(self, gameState):

        def _expectimax(node, depth, a, b, agent):
            """
            Implementation of the expectimax algorithm.
            """
            if depth == 0 or node.isOver():
                action = node.getAgentState(self.index).getDirection()
                return self.evaluate(node), Directions.STOP

            actions = node.getLegalActions(agent)
            random.shuffle(actions)
            if agent == self.index:
                actions.remove(Directions.STOP)

            childNodes = self.opponents if agent == self.index else [self.index]

            best_value, best_action = float('-inf'), None

            if agent == self.index:
                # We are at play
                # We want to find the action that maximizes our score
                for child in childNodes:
                    for action in actions:
                        successor = node.generateSuccessor(agent, action)
                        successor.prevAction = node.getAgentState(agent).getDirection()
                        exp_value = _expectimax(successor, depth - 1, a, b, child)[0]
                        if exp_value > best_value:
                            best_value = exp_value
                            best_action = action
                        a = max(a, best_value)
                        if a >= b:
                            break

                return best_value, best_action
            else:
                # Baddie is at play
                # Since we don't know the opponents strategy, we use
                # probability to chance it for us
                prob = 1.0 / len(actions)
                value = 0.0
                for action in actions:
                    successor = node.generateSuccessor(agent, action)
                    successor.prevAction = node.getAgentState(agent).getDirection()
                    exp_value = _expectimax(successor, depth - 1, a, b, self.index)[0]
                    value += prob * exp_value
                return value, Directions.STOP

        return _expectimax(gameState, 4, float('-inf'), float('inf'), self.index)[1]

    def qLearning(self, gameState):
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        action = None

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon)
            if prob:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(gameState)
        # self.updateWeights(gameState, action)
        return action

    def getPolicy(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)

        if len(legalActions) == 0:
            return None
        else:
            values = []
            for action in legalActions:
                q = self.getQValue(gameState, action)
                # numCarrying = gameState.getAgentState(self.index).numCarrying
                # if numCarrying:
                #     nextState = gameState.generateSuccessor(self.index, action)
                #     if self.getClosestMidPoint(nextState) < self.getClosestMidPoint(gameState):
                #         q *= float('1.00%d' % numCarrying)

                values.append((q[0], action))
        return max(values)[1]

    def getValue(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        try:
            return max(self.getQValue(gameState, action)[0] for action in legalActions)
        except ValueError:
            return 0.0

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return features * self.weights, features

    def getFeatures(self, gameState, action):
        from capture import SCARED_TIME

        features = util.Counter()
        nextState = gameState.generateSuccessor(self.index, action)
        myPos = nextState.getAgentPosition(self.index)

        # Level Progress
        targetFood = self.getFood(nextState).asList()
        levelProgress = (len(self.initialFood) - len(targetFood)) / len(self.initialFood)
        # features['level-progress'] = levelProgress

        # Powerpill
        scared = max(
            nextState.getAgentState(opponent).scaredTimer for opponent in self.opponents)
        if scared:
            features['powerpill'] = 1 - (SCARED_TIME - scared) / SCARED_TIME
        else:
            features['powerpill'] = 0.0

        # Food
        closestFoodDist = self.getClosestFood(gameState, nextState)
        closestFood = (self.maxPathLength - closestFoodDist) / self.maxPathLength
        features['food'] = closestFood

        # Ghost
        closestGhostDist = self.getClosestGhost(gameState, nextState)
        closestGhost = (self.maxPathLength - closestGhostDist) / self.maxPathLength
        features['ghost'] = closestGhost

        # Scared Ghost
        if features['powerpill']:
            closestScaredGhostDist = self.getClosestScaredGhost(gameState, nextState)
            closestScaredGhost = (self.maxPathLength - closestScaredGhostDist) / self.maxPathLength
            features['scared'] = closestScaredGhost
        else:
            features['scared'] = 0.0

        # Entrapment
        deadends = [path for path in self.deadendPaths if myPos in path]
        if deadends and self.notOnMySide(gameState, myPos):
            entrapment = len(deadends[0])
        else:
            entrapment = self.maxEntrapmentLength
        entrapment = (self.maxEntrapmentLength - entrapment) / self.maxEntrapmentLength
        features['entrapment'] = entrapment

        # Action
        currentDirection = gameState.getAgentState(self.index).getDirection()
        features['reverse-action'] = int(currentDirection == Actions.reverseDirection(action))

        # Safe
        closestMidDist = self.getClosestMidPoint(nextState)
        closestMid = (self.maxSafeLength - closestMidDist) / self.maxSafeLength
        # features['safe'] = closestMid

        # Returned
        numCarrying = gameState.getAgentState(self.index).numCarrying
        numCarrying = numCarrying / len(self.initialFood)
        features['risk'] = numCarrying * closestMid
        # if self.index == 0:
        #     print features['risk']

        for k, v in features.iteritems():
            if v in (float('inf'), float('-inf')):
                features[k] = 0.0

        return features

    def getClosestFood(self, gameState, nextState):
        myPos = nextState.getAgentPosition(self.index)
        targetFood = self.getFood(gameState).asList()

        if not targetFood:
            return 0

        return min(self.getMazeDistance(myPos, food) for food in targetFood)

    def getClosestGhost(self, gameState, nextState):
        myPos = nextState.getAgentPosition(self.index)
        closestGhostDist = float('inf')

        for opponent in self.opponents:
            if not gameState.getAgentState(opponent).isPacman:
                enemyPos = gameState.getAgentPosition(opponent)
                if enemyPos is not None:
                    # enemyDist = self.getMazeDistance(myPos, enemyPos)
                    enemyDist = manhattanDistance(myPos, enemyPos)
                    if enemyDist < closestGhostDist:
                        closestGhostDist = enemyDist
        return closestGhostDist

    def getClosestScaredGhost(self, gameState, nextState):
        myPos = nextState.getAgentPosition(self.index)
        closestGhostDist = float('inf')
        for opponent in self.opponents:
            enemyState = gameState.getAgentState(opponent)
            if not enemyState.isPacman and enemyState.scaredTimer > 0:
                enemyPos = gameState.getAgentPosition(opponent)
                if enemyPos is not None:
                    enemyDist = self.getMazeDistance(myPos, enemyPos)
                    if enemyDist < closestGhostDist:
                        closestGhostDist = enemyDist
        return closestGhostDist

    def getClosestMidPoint(self, gameState):
        myPos = gameState.getAgentPosition(self.index)

        if myPos[0] == self.mid:
            return 0

        onRedSide = gameState.isRed(myPos)
        if not (self.red ^ onRedSide):
            return self.maxSafeLength

        safeDistance = float('inf')
        for y in range(self.height):
            midPos = (self.mid, y)
            if midPos in self.legalPositions:
                midDist = self.getMazeDistance(myPos, midPos)
                if midDist < safeDistance:
                    safeDistance = midDist
        return safeDistance

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = gameState.generateSuccessor(self.index, action)
        # currentDirection = gameState.getAgentState(self.index).getDirection()

        # if not gameState.getAgentState(self.index).isPacman:
        #     features['food'] /= 2

        # Calculate the reward
        if (self.gotEaten(gameState, nextState)
            or (features['ghost'] >= 0.98 and features['powerpill'] == 0)):
            reward = -1
            # features['food'] /= 2
        elif self.ateScaredGhost(gameState, nextState):
            reward = 0.2
        elif self.ateFood(gameState, nextState):
            reward = 0.12
        elif self.atePowerpill(gameState, nextState):
            reward = 0.15
        elif features['reverse-action']:
            reward = -0.0025
        else:
            # reward = -1  # time punishment
            reward = 0

        scoreBonus = (nextState.getScore() - gameState.getScore()) / len(self.initialFood)
        reward += scoreBonus

        # if scoreBonus:
        #     features['risk'] = 1.0

        weights = self.weights
        temporalDistance = (reward + self.discountFactor * self.getValue(nextState)) - self.getQValue(gameState, action)[0]
        for feature in features:
            weights[feature] += self.alpha * temporalDistance * features[feature]
        self.weights = weights

    def gotEaten(self, gameState, nextState):
        agentState = nextState.getAgentState(self.index)
        return agentState.configuration == agentState.start

    def ateScaredGhost(self, gameState, nextState):
        myPos = nextState.getAgentPosition(self.index)
        for opponent in self.opponents:
            enemyState = gameState.getAgentState(opponent)
            if not enemyState.isPacman and enemyState.scaredTimer > 0:
                enemyPos = gameState.getAgentPosition(opponent)
                if enemyPos is not None:
                    enemyDist = self.getMazeDistance(myPos, enemyPos)
                    if enemyDist == 0:
                        return True
        return False

    def ateFood(self, gameState, nextState):
        myPos = nextState.getAgentPosition(self.index)
        food = self.getFood(gameState).asList()
        return myPos in food

    def atePowerpill(self, gameState, nextState):
        myPos = nextState.getAgentPosition(self.index)
        powerpills = gameState.getCapsules()
        return myPos in powerpills

    # Defense

    def evaluate(self, gameState):
        myPos = gameState.getAgentPosition(self.index)

        # Get the most likely enemy distances.
        enemyDistances = self.enemyDistances(gameState)

        # Get the pacman on our side.
        invaders = [a for a in self.opponents if
                    gameState.getAgentState(a).isPacman]

        # Get the distance to the pacman and find the minimum.
        pac_distances = [dist for id, dist in enemyDistances if
                         gameState.getAgentState(id).isPacman]
        minPacDistances = min(pac_distances) if len(pac_distances) else 0

        # Get min distance to a power pill.
        capsules = self.getCapsulesYouAreDefending(gameState)
        capsulesDistances = [self.getMazeDistance(myPos, capsule) for capsule in
                             capsules]
        minCapsuleDistance = min(capsulesDistances) if len(capsulesDistances) else 0

        return -999999 * len(invaders) - 10 * minPacDistances - minCapsuleDistance

    def enemyDistances(self, gameState):
        """
        If we are getting a reading for the agent distance then we will return
        this exact distance. In the case that the agent is beyond our sight
        range we will assume that the agent is in the position where our
        belief is the highest and return that position. We will then get the
        distances from the agent to the enemy.
        """
        myPos = gameState.getAgentPosition(self.index)
        dists = []
        for enemy in self.opponents:
            enemyPos = self.lookAhead(gameState, enemy, 1)
            dists.append((enemy, self.getMazeDistance(myPos, enemyPos)))
        return dists

    def lookAhead(self, gameState, agent, n):

        def getVec():
            return {
                Directions.NORTH: (0, m),
                Directions.SOUTH: (0, -m),
                Directions.EAST:  (m, 0),
                Directions.WEST:  (-m, 0),
                Directions.STOP:  (0, 0),
            }[agentDir]

        agentState = gameState.getAgentState(agent)
        agentPos = agentState.getPosition()
        agentDir = agentState.getDirection()

        for m in xrange(n, -1, -1):
            vec = getVec()
            x = agentPos[0] + vec[0]
            y = agentPos[1] + vec[1]
            pos = (x, y)
            if pos in self.legalPositions:
                return pos


class OffensiveAgent(BaseAgent):
    # TODO: detect if I can get cornered, and if I can then check how far the
    # closest opponent is

    def registerInitialState(self, gameState):
        return super(OffensiveAgent, self).registerInitialState(gameState)

        self.mode = 'attack'

    def chooseAction(self, gameState):
        # If I am already on my side, and there is a baddie on my side
        # go defensive
        # otherwise be offensive

        # Check if the enemy has any pacman.
        # invaders = [a for a in self.opponents if
        #             gameState.getAgentState(a).isPacman]
        # numInvaders = len(invaders)

        # # If there are no pacman on our side or the poison pill is active we
        # # should act like an offensive agent.
        # if numInvaders > 0 and not gameState.getAgentState(self.index).isPacman:
        #     return super(OffensiveAgent, self).expectimax(gameState)
        # else:
        return super(OffensiveAgent, self).qLearning(gameState)

    # def final(self, gameState):
    #     write_weights(self.weights)


class DefensiveAgent(BaseAgent):
    """
    This is a defensive agent that likes to attack. If there are no enemy pacman
    then the defensive agent will act on the offensive agent evaluation function.
    We do not use carry limits though because the agent will retreat when the
    other team has a pacman.
    """

    def registerInitialState(self, gameState):
        super(DefensiveAgent, self).registerInitialState(gameState)

        self.mode = 'defense'

        self.epsilon = 0.0
        self.alpha = 0.0

    def chooseAction(self, gameState):
        # If there are no baddies on my side
        # go offensive

        # Check if the enemy has any pacman.
        invaders = [a for a in self.opponents if
                    gameState.getAgentState(a).isPacman]
        numInvaders = len(invaders)

        # Check if we have the poison active.
        scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in
                         self.opponents]

        # If there are no pacman on our side or the poison pill is active we
        # should act like an offensive agent.
        if numInvaders == 0 or min(scaredTimes) > 8:
            return super(DefensiveAgent, self).qLearning(gameState)
        else:
            return super(DefensiveAgent, self).expectimax(gameState)
