#!/usr/bin/env python3
#
#   RRT_Star.py
#
from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
import random
import math

from planarutils import *
from sklearn.neighbors import KDTree


######################################################################
#
#   General/World Definitions
#
#   List of objects, start, goal, and parameters.
#
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)

# obstacles = ()

# obstacles = ((( 2, 6), ( 3, 2), ( 4, 6)),
#              (( 6, 5), ( 7, 7), ( 8, 5)),
#              (( 6, 9), ( 8, 9), ( 8, 7)),
#              ((10, 3), (11, 6), (12, 3)))

obstacles = (((6, 4.1), (6, 8), (8, 8), (8, 4.1)), 
             ((6, 2), (6, 3.9), (8, 3.9), (8, 2)))

(startx, starty) = ( 1, 5)
(goalx,  goaly)  = (13, 5)

dstep = 0.25
Nmax  = 1000


######################################################################
#
#   Visualization
#
class Visualization:
    def __init__(self):
        # Clear and show.
        self.ClearFigure()
        self.ShowFigure()

    def ClearFigure(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the obstacles.
        for obst in obstacles:
            for i in range(len(obst)):
                if i == len(obst) - 1:
                    next = 0
                else:
                    next = i + 1
                plt.plot((obst[i][0], obst[next][0]), (obst[i][1], obst[next][1]), 'k-', linewidth=2)

    def ShowFigure(self):
        # Show the plot.
        plt.pause(0.001)


######################################################################
#
#   State Definition
#
class State:
    def __init__(self, x, y):
        # Remember the (x,y) position.
        self.x = x
        self.y = y


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Point %2d,%2d>" % (self.x, self.y))

    # Draw where the state is:
    def Draw(self, *args, **kwargs):
        plt.plot(self.x, self.y, *args, **kwargs)
        plt.pause(0.001)

    # Return a tuple of the coordinates.
    def Coordinates(self):
        return (self.x, self.y)


    ############################################################
    # RRT_Star Functions:
    # Compute the relative distance to another state.    
    def DistSquared(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)

    # Compute/create an intermediate state.
    def Intermediate(self, other, alpha):
        return State(self.x + alpha * (other.x - self.x),
                     self.y + alpha * (other.y - self.y))

    # Check the local planner - whether this connects to another state.
    def ConnectsTo(self, other):
        for obst in obstacles:
            if SegmentCrossBox(((self.x, self.y), (other.x, other.y)),
                                    obst):
                return False
        return True


######################################################################
#
#   Tree Node Definition
#
#   Define a Node class upon which to build the tree.
#
class Node:
    def __init__(self, state, parentnode, draw=True):
        # Save the state matching this node.
        self.state = state

        # Link to parent for the tree structure.
        self.parent = parentnode

        # Determine a cost to reach a certain node by adding onto the distance to reach the parent node
        if (parentnode == None): # if node is the starting node, it will have no parent, so the cost to reach it will default to 0
            self.creach = 0
        else:
            self.creach = parentnode.creach + math.sqrt(state.DistSquared(parentnode.state))

        if draw:
            self.Draw('r-', linewidth=1)

    # Draw a line to the parent.
    def Draw(self, *args, **kwargs):
        if self.parent is not None:
            plt.plot((self.state.x, self.parent.state.x),
                     (self.state.y, self.parent.state.y),
                     *args, **kwargs)
            plt.plot(self.state.x, self.state.y, 'go', markersize=2)
            plt.pause(0.001)

#
#   Connect the nearest neighbors
#
def KNearestNeighbors(tree, K):
    # Determine the indices for the nearest neighbors.  This also
    # reports the node itself as the closest neighbor, so add one
    # extra here and ignore the first element below.
    X = np.array([node.state.Coordinates() for node in tree])
    kdt = KDTree(X)
    dist, idx = kdt.query(X, k=(K+1)) # adding one since the search tree includes the point itself
    dist = dist[len(dist)-1,1:] # getting rid of the first point when reporting nearest neighbors
    idx = idx[len(idx)-1,1:]
    return (dist, idx)
    


######################################################################
#
#   RRT_Star Functions
#
#   Again I am distiguishing state (containing x/y information) and
#   node (containing tree structure/parent information).
#

def K(tree):
    return math.floor(2 * math.e * math.log(len(tree)))

def sample(startstate, goalstate, max_cost):
    if max_cost != np.Infinity:
        found = False
        cmin = np.sqrt(startstate.DistSquared(goalstate))
        midpoint = ((startstate.x + goalstate.x) / 2, (startstate.y + goalstate.y) / 2)
        while not found:
            state = State(random.uniform(startstate.x - ((max_cost - cmin)/2), goalstate.x + ((max_cost - cmin)/2)),
                          random.uniform(startstate.y - np.sqrt(max_cost**2 - cmin**2)/2, startstate.y + np.sqrt(max_cost**2 - cmin**2)/2))
            if ((state.x - midpoint[0]) ** 2 ) / (max_cost / 2)**2 + ((state.y - midpoint[1]) ** 2) / (np.sqrt(max_cost**2 - cmin**2)/2)**2 <= 1:
                found = True
        return state
    else:
        return State(random.uniform(xmin, xmax),
                     random.uniform(ymin, ymax))

def draw_ellipse(x, y, a, b):
    t = np.linspace(0, 2*math.pi, 100)
    plt.plot( x+a*np.cos(t) , y+b*np.sin(t), 'b-', linewidth=2)
    plt.pause(0.001)


def RRT_Star(tree, startstate, goalstate, Nmax):
    sols = []
    best_sol = 0
    iters = 0
    while True:
        # Determine the target state.
        if len(sols) == 0:
            cmax = np.Infinity
        else:
            cmax = sols[0].creach
        targetstate = sample(tree[0].state, goalstate, cmax)
        # Find the nearest node (node with state nearest the target state).
        # This is inefficient (slow for large trees), but simple.
        list = [(node.state.DistSquared(targetstate), node) for node in tree]
        (d2, nearestnode)  = min(list)
        d = np.sqrt(d2)
        neareststate = nearestnode.state

        # Determine the next state, a step size (dstep) away.

        newstate = State(neareststate.x + dstep * (targetstate.x - neareststate.x) / np.sqrt(neareststate.DistSquared(targetstate)),
                          neareststate.y + dstep * (targetstate.y - neareststate.y) / np.sqrt(neareststate.DistSquared(targetstate)))

        # Check whether to attach (creating a new node).
        if neareststate.ConnectsTo(newstate):
            newnode = Node(newstate, nearestnode, draw=False)
            tree.append(newnode)
            k = K(tree)
            if (len(tree) > k):
                (dist, idx) = KNearestNeighbors(tree, k)

                # check each of the newnode's nearest neighbors if they can create a better path to newnode
                newnear = nearestnode
                mincost = newnode.creach
                for i in idx:
                    nearnode = tree[i]
                    if nearnode.state.ConnectsTo(newstate) and nearnode.creach + math.sqrt(newstate.DistSquared(nearnode.state)) < mincost:
                        # print("SHORTER PATH!")
                        newnear = nearnode
                        mincost = nearnode.creach + math.sqrt(newstate.DistSquared(nearnode.state))
                tree[-1].parent = newnear
                tree[-1].creach = mincost
                tree[-1].Draw('r-', linewidth=1)

                # check if a new path can be made through xnew to one of its nearest neighbors
                for j in idx:
                    nearnode = tree[i]
                    if nearnode.state.ConnectsTo(newstate) and nearnode.creach + math.sqrt(newstate.DistSquared(nearnode.state)) < nearnode.creach:
                        print("REWIRED")
                        nearnode.parent = newnode
                
            # Also try to connect the goal.
            if np.sqrt(newstate.DistSquared(goalstate)) < 2*dstep and newstate.ConnectsTo(goalstate):
                goalnode = Node(goalstate, newnode)
                node = goalnode
                while node.parent is not None:
                    node.Draw('b-', linewidth=2)
                    plt.plot(node.state.x, node.state.y, 'co', markersize=3)
                    node = node.parent
                sols.append(goalnode)
                sols.sort(key=lambda x: x.creach)
                print("Solution found with cost =", goalnode.creach, "!")
                if best_sol == sols[0]:
                        iters += 1
                else:
                    best_sol = sols[0]
                if (goalnode.creach - np.sqrt(startstate.DistSquared(goalstate)) < 0.001 or iters > 5):
                    plt.cla()
                    Visual = Visualization()
                    # Show the start/goal states.
                    startstate.Draw('ro')
                    goalstate.Draw('ro')
                    Visual.ShowFigure()
                    node = best_sol
                    while node.parent is not None:
                        node.Draw('b-', linewidth=2)
                        plt.plot(node.state.x, node.state.y, 'co', markersize=3)
                        node = node.parent
                    return best_sol
                else:
                    draw_ellipse((startx + goalx) / 2, (starty + goaly) / 2, goalnode.creach / 2, np.sqrt(goalnode.creach ** 2 - startstate.DistSquared(goalstate)) / 2)
                    print("The next cost upper bound is", sols[0].creach)
                    input("Now sampling in informed subset\n")
                    plt.cla()
                    tree = tree[:1]
                    # Create the figure.
                    Visual = Visualization()
                    # Show the start/goal states.
                    startstate.Draw('ro')
                    goalstate.Draw('ro')
                    Visual.ShowFigure()
                    draw_ellipse((startx + goalx) / 2, (starty + goaly) / 2, best_sol.creach / 2, np.sqrt(best_sol.creach ** 2 - startstate.DistSquared(goalstate)) / 2)


        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            return None


######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with step size ', dstep, ' and up to ', Nmax, ' nodes.')

    # Create the figure.
    Visual = Visualization()


    # Set up the start/goal states.
    startstate = State(startx, starty)
    goalstate  = State(goalx,  goaly)
    
    # Show the start/goal states.
    startstate.Draw('ro')
    goalstate.Draw('ro')
    Visual.ShowFigure()
    input("Showing basic world (hit return to continue)")


    # Start the tree with the start state and no parent.
    tree = [Node(startstate, None)]

    # Execute the search (return the goal leaf node).
    node = RRT_Star(tree, startstate, goalstate, Nmax)

    # Check the outcome
    if node is None:
        print("UNABLE TO FIND A PATH in", Nmax, "steps")
        input("(hit return to exit)")
        return
    
    
    print("PATH found after", len(tree),"samples")
    input("Press enter to exit")
    return


if __name__== "__main__":
    main()
