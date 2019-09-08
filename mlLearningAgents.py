# practiceAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

from pacman import Directions
from game import Agent
import random
import game
import util
from util import Counter
import random
import inspect
import itertools

# QLearnAgent
#
class QLearnAgent(Agent):
    """"
    An exploratory SARSA agent that actively learns Q values for each state-action pair in the Pacman smallGrid
    environment. This implementation has been largely adapted from the Q-Learning pseudo-code algorithm on p.860 of
    Artificial Intelligence A Modern Approach (P. Norvig & S. Russell) and then altered to reflect the update
    behaviour of SARSA such that Q values are updated at the *end* of the s,a,r,s',a' quintuple.

    The Q-learning algo was adapted to SARSA due to the presence of an adversarial ghost agent, meaning it's 'better to
    learn a Q-function for what will actually happen rather than what the agent would like to happen.' [AIMA, p.861]
    """
    # Constructor used to initialise a number of variables
    def __init__(self, alpha=0.2, gamma=0.8, numTraining = 10, Ne=5, rPlus=10):

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.trainFlag = True
        # Number of times necessary to visit a state before counting it as explored, determined experimentally as 5
        self.Ne = Ne
        # Optimistic reward to be returned until the state is sufficiently explored
        self.rPlus = rPlus

        # util.Counter() objects for storing Q values and counts of state visits; has the useful behaviour over standard
        # dictionaries for this purpose of automatically initialising a new key with a default value of 0, similar to
        # collections.defaultdict()
        self.Q = Counter()
        self.N_sa = Counter()

        self.s = None
        self.a = None
        # For handling reward calculations between states
        self.prevScore = None

    # All accessor functions used courtesy of Simon Parsons.
    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    def f(self, u, n):
        """
        Controls the exploration and exploitation behaviour of the agent. Checks whether the agent has visited this
        particular state more than 'Ne' times as dictated by the user on startup. If so, the Q value of that state is
        returned - i.e. the learned policy is exploited. Otherwise, an optimistic reward value 'rPlus' is output to
        encourage further exploration.
        """
        if n < self.Ne:
            return self.rPlus
        else:
            return u

    def rPrime(self, state):
        """
        Reward assignment: rewards reflect the difference in score between the last and current game state.
        """
        if self.prevScore is None:
            r_prime = self.prevScore = state.getScore()

        else:
            r_prime = state.getScore() - self.prevScore

        return r_prime

    def Q_update(self, state):
        """
        Determines the best action to take from the agent's current position based on the Q values in neighbouring
        states, then retrospectively updates the Q value of the previous state based on the determined action; an
        'on-policy' update.
        """

        Q, N_sa = self.Q, self.N_sa
        s, a = self.s, self.a
        alpha, gamma = self.alpha, self.gamma

        # retrieve legal actions for current state
        A = state.getLegalPacmanActions()
        if Directions.STOP in A:
            A.remove(Directions.STOP)

        # retrieve current state information corresponding to the locations of pacman, the ghost and food items;
        # type as an immutable tuple for use as a dictionary key later
        s_prime = (state.getPacmanPosition(), tuple(state.getGhostPositions()), str(state.getFood()))

        r_prime = self.rPrime(state)

        # Exploratory behaviour if training
        if self.trainFlag:
            # retrieve Q values for possible eventualities along with number of times each has been experienced
            Q_values = [Q[s_prime, a_prime] for a_prime in A]
            N_values = [N_sa[s_prime, a_prime] for a_prime in A]

            # use f(u, n) to determine whether to follow learned policy for each possible state or to explore it
            # further
            Q_N_values = zip(Q_values, N_values)
            f_values = [self.f(u, n) for u, n in Q_N_values]

            # slightly convoluted implementation of argmax for choosing next action; can't use Counter.argMax() method
            # or index by max index outright as max() in Python 2 always returns index position of the first max value
            # in a list leading to predictable off-policy behaviour. Instead break ties using random.choice()
            indexes = []
            for i,j in enumerate(f_values):
                if j == max(f_values):
                    indexes.append(i)
            argMaxIdx = random.choice(indexes)
            a_prime = A[argMaxIdx]

        # greedy policy exploitation if training has finished
        else:
            Q_values = [Q[s_prime, a_prime] for a_prime in A]
            indexes = []
            for i,j in enumerate(Q_values):
                if j == max(Q_values):
                    indexes.append(i)
            argMaxIdx = random.choice(indexes)
            a_prime = A[argMaxIdx]

        if s is not None:
                # increment N_sa
                N_sa[s, a] += 1  # useful util.Counter() behaviour means don't have to initialise count explicitly

                # Update Q in temporal difference fashion; i.e. use observed transitions to update Q values of observed
                # states.
                # Not necessary to explicitly code different behaviour if no longer exploring as alpha=0 in that case
                Q[s, a] = Q[s, a] + alpha * (r_prime + gamma * Q[s_prime, a_prime] - Q[s, a])

        self.s = s_prime
        self.a = a_prime
        self.prevScore = state.getScore()

        return a_prime

    def getAction(self, state):
        """
        The main method required by the game. Called every time that Pacman is expected to move [Simon Parsons].

        Simply returns the action determined by self.Q_update()
        """

        return self.Q_update(state)

    def final(self, state):
        """
        Ensures the final Q update is recorded at the end of a game, without which pacman will not learn to avoid ghosts
        """

        Q, N_sa = self.Q, self.N_sa
        s, a, = self.s, self.a

        end = "A game just ended!"
        print "%s\nScore: %d"%(end, state.getScore())

        Q[s, a] = state.getScore()-self.prevScore

        self.s = self.a = self.prevScore = None
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes [Simon Parsons]
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off training flag and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.trainFlag = False


