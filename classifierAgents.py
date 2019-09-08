
from __future__ import division
from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
import math


""" 
The following functions and classes define methods necessary for building a decision tree classifier, then 
implementing an ensemble of these trees via bootstrap aggregation in order to return a single classification label
as agreed upon by a maximum vote of the ensemble.
 
Parts of this method were largely adapted from 'Let's Write a Decision Tree Classifier from Scratch', a YouTube video
by Josh Gordon for Google Developers ML tutorial series. 

Namely, the buildTree() function below works using recursive instantiation of 'branch' and 'leaf' classes which 
is an elegant implementation and adapted straight from his example. Other methods were experimented with such as trying
to recursively build dictionaries of nodes but threading state variables through the recursion when the function 
calls itself twice per call was too difficult, so this object-based technique was adapted explicitly. 
 """


''' Splitting criterion '''

def entropy(data):
    """
    Calculate the overall entropy of the input data based on the proportion of classes it contains
    """
    # count classes in 'data' argument
    c = classCounter(data)

    total = sum(c.values())

    # calculate proportion of each class
    classProportions = [c[k] / total for k in c.keys()]

    # calculate subsequent entropies
    entropies = [(-p * math.log(p, 2)) for p in classProportions]

    return sum(entropies)


def infoGain(true, false, H_prev):
    """
    Calculate the decrease in entropy associated with the class proportions in the next level of the tree in
    comparison with the current level
    """
    # relative weights of true and false subsets
    p_T = len(true) / (len(true) + len(false))
    p_F = 1 - p_T

    # information gain
    SA = H_prev - p_T * entropy(true) - p_F * entropy(false)
    return SA


''' Data splitting '''

def partition(colIdx, data):
    """
    Split the input data into two groups depending on the value for each tuple at the specified colIdx position; if
    an entry has '1' for the specified feature, assign to the 'true' group, 'false' if '0'
    """
    true = []
    false = []
    for row in data:
        if row[colIdx] == 1:
            true.append(row)
        else:
            false.append(row)
    return true, false


def bestSplit(data):
    """
    Iterate through attributes of feature vector, calculate information gain associated with splitting the current data
    set based on each feature in turn; return a dictionary containing both the feature index and gain value of the most
    informative feature
    """

    n_attribs = len(data[0]) - 1

    splitOn = {'index': 0, 'gain': 0}

    # entropy of input data before splitting
    currEntropy = entropy(data)

    maxGain = 0

    # iterate through each row of the input data, partitioning the data into 'true' and 'false' groups based on the
    # presence of a 1 or 0 in each index position
    for idx in range(n_attribs):
        true, false = partition(idx, data)

        if len(true) == 0 or len(false) == 0:
            continue

        # determine the information gain associated with splitting the data based on the current index position
        gain = infoGain(true, false, currEntropy)

        # if the info gain is larger than the current value store in maxGain then overwrite its value and store the
        # details of this feature (index position, associated information gain)
        if gain > maxGain:
            maxGain = gain
            splitOn['index'] = idx
            splitOn['gain'] = gain

    return splitOn


''' Classifying '''

def classCounter(data):
    """
    Retrieve the class labels of the input rows, count the number of each class and return a dictionary of counts
    """
    target = np.array([row[-1] for row in data])
    counts = dict()
    for i in range(4):
        if len(target[target == i]) > 0:
            counts[i] = len(target[target == i])

    return counts


def classChooser(classes):
    """
    Takes counts of classes at a leaf node and returns a classification label. If there is no clear majority class,
    class selection is chosen probabilistically based on the proportion of class labels present
    """

    total = sum(classes.values())

    # Retrieve class labels
    C = np.sort(classes.keys())

    # Calculate class proportions
    P = [classes[c] / total for c in C]

    # if there's no clear majority classification, return a weighted random pick of the class labels
    if all(p < 0.5 for p in P):
        return np.random.choice(C, p=P)

    else:
        return C[P.index(max(P))]


def classify(example, node):
    """
    Recursively unpacks a tree object and ultimately returns a classification label based on the attribute values of a
    provided example
    """

    # base case of recursion is if the current node being unpacked is a terminal leaf; if so, return a classification
    # based on the information stored at that point
    if isinstance(node, leaf):
        classes = node.classes

        return classChooser(classes)

    # retrieve the most discriminating feature at the current tree level
    idx = node.featureNum

    # check which child branch node to unpack based on the value of the example for this feature
    if example[idx] == 1:
        return classify(example, node.trueChild)
    else:
        return classify(example, node.falseChild)


''' Tree Building '''

class leaf:
    """
    Used to store counts of labels at a terminal node in the decision tree, formed when the instances
    at that point are completely classified, the information gain associated with splitting further is otherwise zero or
    the maximum depth of the tree has been reached as specified in the arguments for buildTree()
    """
    def __init__(self, data):
        self.classes = classCounter(data)


class branch:
    """
    Represents a splitting node in the decision tree; it stores the index of the feature on which the
    split is being made, as well as references to the resultant branch/leaf instances
    """
    def __init__(self, index, trueChild, falseChild):
        self.featureNum = index
        self.trueChild=trueChild
        self.falseChild=falseChild


def buildTree(data, maxDepth=None, level=None):
    """
    Largely adapted from 'Let's Write a Decision Tree Classifier from Scratch' by Josh Gordon for Google Developers ML
    tutorial series.

    Builds a decision tree recursively, creating a recursive object of heirarchical branch and leaf nodes. Each single
    branch instance in the tree references either a leaf instance or another two branch instances (one for the 'true'
    branching and another for 'false')
    """

    # if a maximum depth has been defined, the base case for recursion is that the level counter is equal to the
    # specified maxDepth, otherwise recursion continues until there is no information to be gained by splitting further
    if maxDepth:

        if not level:
            level = 1

        # if maxDepth is reached, return a terminal node with class counts of the current data subset regardless of
        # the potential information gain from further splits
        if level == maxDepth:
            return leaf(data)

        # determine the current best attribute for splitting the data
        splits = bestSplit(data)

        idx = splits['index']

        # return a leaf node if no further classification possible
        if splits['gain'] == 0:
            return leaf(data)

        # otherwise partition the data into true and false subsets based on the splitting attribute
        true, false = partition(idx, data)

        level += 1

        # then recursively call the buildTree() function on each of these subsets in turn
        trueChild = buildTree(true, maxDepth, level)

        falseChild = buildTree(false, maxDepth, level)

        # before returning a branch node instance referring to these subsequent instances
        return branch(idx, trueChild, falseChild)

    # behaviour if no maxDepth
    else:
        splits = bestSplit(data)

        idx = splits['index']

        if splits['gain'] == 0:
            return leaf(data)

        true, false = partition(idx, data)

        trueChild = buildTree(true)

        falseChild = buildTree(false)

        return branch(idx, trueChild, falseChild)


''' Ensemble methods '''


def bagging(data, n_trees, maxDepth=None, sampleSize=0.8):
    """
    Bootstrap aggregation for creating an ensemble of decision tree items which can be stored and used for making
    majority-vote classifications with improved accuracy versus one tree alone
    """

    # create a container for the multiple trees
    bag = []

    for n in range(n_trees):

        # create training data samples of size sampleSize from the original data via sampling with replacement
        idxs = range(int(sampleSize * len(data)))
        sampleIdx = [np.random.choice(idxs) for idx in idxs]
        data = np.array(data)
        sample = data[sampleIdx]

        # train a new tree on each random sample in turn and append the tree to the bag container
        dtree = buildTree(sample, maxDepth=maxDepth)
        bag.append(dtree)

    return bag


def majorityVote(example, bag):
    """
    Based on an ensemble of trees, classify an example data point using each model in the ensemble, then return the
    majority consensus of all models on that classification
    """
    votes = [classify(example, dtree) for dtree in bag]

    # return modal class
    return max(set(votes), key=votes.count)


class ClassifierAgent(Agent):
    """
    An agent subclass that returns an action based on the output of a classifier system
    """

    def __init__(self):
        print "Initialising"

    def convertToArray(self, numberString):
        """
        Used courtesy of Simon Parsons.
        Take a string of digits and convert to an array of numbers. Exploits the fact
        that we know the digits are in the range 0-4.
        """
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    def registerInitialState(self, state):
        """
        Used courtesy of Simon Parsons. Modified to keep target attribute in same
        array as features.
        """

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray)):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)

        # train an ensemble of decision trees based on the training data
        self.classifier = bagging(self.data, 11)

    def convertNumberToMove(self, number):
        """
        Used courtesy of Simon Parsons.
        Turn the numbers from the feature set into actions
        """
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    def getAction(self, state):

        features = api.getFeatureVector(state)

        # Return majority classification from ensemble
        moveNumber = majorityVote(features, self.classifier)

        legal = api.legalActions(state)

        return api.makeMove(self.convertNumberToMove(moveNumber), legal)

