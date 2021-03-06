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
from matplotlib.patches import Ellipse


######################################################################
#
#   General/World Definitions
#
#   List of objects, start, goal, and parameters.
#
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)

(startx, starty) = ( 1, 5)
(goalx,  goaly)  = (13, 5)

# (startx, starty) = (random.randint(xmin, xmax), random.randint(ymin, ymax))

# (startx, starty) = (7, 5)
# (goalx, goaly) = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
# while (goalx, goaly) == (startx, starty):
#     (goalx, goaly) = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))

dstep = 0.25
Nmax  = 2000

# Generates random obstacles
def generateObstacles():
    obstacles = []

    num_obstacles = random.randint(50, 100)

    # print(num_obstacles)
    while len(obstacles) < num_obstacles:
        triangle = []
        while len(triangle) < 3:
            i = len(triangle)
            size = 1
            if i == 0:
                point = (random.uniform(xmin, xmax), 
                         random.uniform(ymin, ymax))
            elif i == 1:
                point = (triangle[i - 1][0] + random.uniform(0, size),
                         triangle[i - 1][1] + random.uniform(0, size))
            elif i == 2:
                point = \
                  (random.uniform(triangle[0][0] - size, triangle[1][0] + size),
                   random.uniform(triangle[0][1] - size, triangle[1][1] + size))
            if point in triangle or point == (startx, starty) \
                or point == (goalx, goaly):
                continue
            triangle.append(point)
        if PointInTriangle((startx, starty), tuple(triangle)) or \
           PointInTriangle((goalx, goaly), tuple(triangle)):
               continue
        obstacles.append(tuple(triangle))

    return tuple(obstacles)

# obstacles = ()

obstacles = ((( 2, 6), ( 3, 2), ( 4, 6)),
             (( 6, 5), ( 7, 7), ( 8, 5)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))


# obstacles = generateObstacles()
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
            if SegmentCrossTriangle(((self.x, self.y), (other.x, other.y)),
                                    obst):
            # if SegmentCrossBox(((self.x, self.y), (other.x, other.y)),
            #                         obst):
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

    # Update creach
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
    rand = random.random()
    if max_cost != np.Infinity:
        if rand < 0.05:
            return goalstate
        found = False
        cmin = np.sqrt(startstate.DistSquared(goalstate))
        (h, k) = ((startstate.x + goalstate.x) / 2, (startstate.y + goalstate.y) / 2)
        a = max_cost / 2
        b = np.sqrt(max_cost**2 - cmin**2) / 2
        angle = np.arctan2((goalstate.y - startstate.y),(goalstate.x - startstate.x))
        if angle % np.pi == 0:
            ellminx = h - a
            ellmaxx = h + a
            ellminy = k - b
            ellmaxy = k + b
        elif angle % np.pi / 2 == 0:
            ellminx = h - b
            ellmaxx = h + b
            ellminy = k - a
            ellmaxy = k + a
        else:
            x1 = h - np.sqrt(a**2 + b**2 + (a**2 - b**2) * np.cos(2*angle)) / np.sqrt(2)
            x2 = h + np.sqrt(a**2 + b**2 + (a**2 - b**2) * np.cos(2*angle)) / np.sqrt(2)
            y1 = k - ((((a**2 - b**2)**2 * (-a**2 - b**2 + (a**2 - b**2) * np.cos(2*angle))) / (np.cos(4*angle) - 1)) * np.sin(2*angle)) / (a**2 - b**2)
            y2 = k + ((((a**2 - b**2)**2 * (-a**2 - b**2 + (a**2 - b**2) * np.cos(2*angle))) / (np.cos(4*angle) - 1)) * np.sin(2*angle)) / (a**2 - b**2)
            ellminx = min(x1, x2)
            ellmaxx = max(x1, x2)
            ellminy = min(y1, y2)
            ellmaxy = max(y1, y2)

        while not found:
            state = State(random.uniform(ellminx, ellmaxx),
                          random.uniform(ellminy, ellmaxy))
            if (((state.x - h)*np.cos(angle) + (state.y - k)*np.sin(angle)) ** 2) / a**2 + \
               (((state.x - h)*np.sin(angle) - (state.y - k)*np.cos(angle)) ** 2) / b**2 <= 1:
                found = True
        return state
    else:
        if rand < 0.15:
            return goalstate
        return State(random.uniform(xmin, xmax),
                     random.uniform(ymin, ymax))

def draw_ellipse(s, g, a, b):
    mid = ((s[0] + g[0]) / 2, (s[1] + g[1]) / 2)
    angle = np.arctan2(g[1]-s[1], g[0]-s[0]) * 180 / np.pi
    ell = Ellipse(mid, a, b, angle=angle, alpha=0.5)
    plt.gca().add_artist(ell)
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
            # print(len(tree))
            # print(k)
            (dist, idx) = KNearestNeighbors(tree, min(k, len(tree)-1))

            # check each of the newnode's nearest neighbors if they can create a better path to newnode
            newnear = nearestnode
            mincost = newnode.creach
            for i in idx:
                nearnode = tree[i]
                if nearnode.state.ConnectsTo(newstate) and nearnode.creach + math.sqrt(newstate.DistSquared(nearnode.state)) < mincost:
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
                    print("PATH COST NOT REDUCED. Reverting back to last best solution...")
                else:
                    best_sol = sols[0]
                    iters = 0
                    print("PATH COST REDUCED!")
                if (goalnode.creach - np.sqrt(startstate.DistSquared(goalstate)) < 0.001 or iters >= 3):
                    return best_sol
                else:
                    draw_ellipse((startx, starty), (goalx, goaly), goalnode.creach, np.sqrt(goalnode.creach ** 2 - startstate.DistSquared(goalstate)))
                    print("The next cost upper bound is", sols[0].creach)
                    input("Now sampling in informed subset\n")
                    plt.cla()
                    tree = tree[:1]
                    # Create the figure.
                    Visual = Visualization()
                    # Show the start/goal states.
                    startstate.Draw('ro')
                    goalstate.Draw('bo')
                    Visual.ShowFigure()
                    draw_ellipse((startx, starty), (goalx, goaly), best_sol.creach, np.sqrt(best_sol.creach ** 2 - startstate.DistSquared(goalstate)))


        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            if len(sols) == 0:
                return None
            return best_sol

def PostProcess(gnode):
    start = gnode # the "bottom" of the path
    curr = gnode  # the current node you are considering
    prev = gnode  # the previous "good" node that can connect
   
    while curr.parent is not None:
        while curr is not None and curr.state.ConnectsTo(start.state):
            prev = curr
            curr = prev.parent
        start.parent = prev
        
        # connect the rest of the paths
        start = prev
        curr = prev


def RRT_Rect(tree, startstate, goalstate, Nmax, h, path):
    while True:
        # Tube sampling
        targetstate = State(random.uniform(xmin, xmax), 
                            random.uniform(ymin, ymax))

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
            # Check whether newstate is within our segment tube
            for node in path:
                if PointNearSegment(h, newstate.Coordinates(), \
                        (node.state.Coordinates(), \
                         node.parent.state.Coordinates())):
                    newnode = Node(newstate, nearestnode, draw=False)
                    tree.append(newnode)
                    k = K(tree)
                    
                    # Max is stupid
                    (dist, idx) = KNearestNeighbors(tree, min(k, len(tree)-1))
                    newnear = nearestnode
                    mincost = newnode.creach
                    for i in idx:
                        nearnode = tree[i]
                        if nearnode.state.ConnectsTo(newstate) and \
                            nearnode.creach + \
                            math.sqrt(newstate.DistSquared(nearnode.state)) < mincost:
                            newnear = nearnode
                            mincost = nearnode.creach + math.sqrt(newstate.DistSquared(nearnode.state))
                    tree[-1].parent = newnear
                    tree[-1].creach = mincost
                    tree[-1].Draw('r-', linewidth=1)

                    # Also try to connect the goal.
                    if np.sqrt(newstate.DistSquared(goalstate)) < 2*dstep and \
                            newstate.ConnectsTo(goalstate):
                        goalnode = Node(goalstate, newnode)
                        return goalnode
                    break

        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            return 
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
    goalstate.Draw('bo')
    Visual.ShowFigure()
    input("Showing basic world (hit return to continue)")


    # Start the tree with the start state and no parent.
    tree = [Node(startstate, None)]

    # Execute the search (return the goal leaf node).
    gnode = RRT_Star(tree, startstate, goalstate, Nmax)
    

    # Check the outcome
    if gnode is None:
        print("UNABLE TO FIND A PATH in", Nmax, "steps")
        input("(hit return to exit)")
        return
    
    plt.cla()
    Visual = Visualization()
    # Show the start/goal states.
    startstate.Draw('ro')
    goalstate.Draw('bo')
    Visual.ShowFigure()
    node = gnode
    cost = node.creach
    while node.parent is not None:
        node.Draw('b-', linewidth=2)
        plt.plot(node.state.x, node.state.y, 'co', markersize=3)
        node = node.parent
    print("PATH found after", len(tree),"samples with cost =", cost)
    input("Press enter to post-process")
    
    # Post Processing
    # Sees if next node could be connected
    best_sol = gnode 
    best_path = []
    fails = 0
    while True:
        PostProcess(gnode)
        plt.cla()
        Visual = Visualization()
        # Show the start/goal states.
        startstate.Draw('ro')
        goalstate.Draw('bo')
        Visual.ShowFigure()
        node = gnode
        path = []
        while node.parent is not None:
            path.append(node)
            node.Draw('g-', linewidth=2)
            plt.plot(node.state.x, node.state.y, 'co', markersize=3)
            node = node.parent
        if gnode.creach <= best_sol.creach:
            best_sol = gnode
            best_path = path
            print('NEW BEST SOLUTION COST: ', best_sol.creach)
        else:
            fails += 1
            print('WORSE PATH FOUND (cost: ', gnode.creach, ') REVERTING TO OPTIMAL SOLUTION')

        print("POST PROCESSED PATH") 
        input('Press enter to resample the optimal path')

        if fails > 3:
            print("PATH COST NOT REDUCED. Reverting back to last best solution...")
            print('BEST SOL: ', best_sol.creach)
            plt.cla()
            Visual = Visualization()
            # Show the start/goal states.
            startstate.Draw('ro')
            goalstate.Draw('bo')
            Visual.ShowFigure()
            node = best_sol 
            while node.parent is not None:
                node.Draw('g-', linewidth=2)
                plt.plot(node.state.x, node.state.y, 'co', markersize=3)
                node = node.parent
            input('Press enter to quit...')
            return

        # "Tube" resampling of post-processed path
        r = 0.25 # height of rectangle 
        # Restart the tree
        tree = [Node(startstate, None)]
        gnode = RRT_Rect(tree, startstate, goalstate, Nmax, r, best_path)
        plt.cla()
        Visual = Visualization()
        # Show the start/goal states.
        startstate.Draw('ro')
        goalstate.Draw('bo')
        Visual.ShowFigure()
        node = gnode
        while node.parent is not None:
            path.append(node)
            node.Draw('b-', linewidth=2)
            plt.plot(node.state.x, node.state.y, 'co', markersize=3)
            node = node.parent
        print('RESAMPLED TUBE PATH')
        input('Press enter to post process the tube path')
     
if __name__== "__main__":
    main()
