# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from queue import PriorityQueue
from queue import deque
import heapq
import queue
import copy

# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): DISTANCE(i, j)
                for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    final_path=[]  
    #the path we will return
    queue=[]
    queue.append(maze.start)
    #add the start of maze to queue
    visited=set()

    while queue:
        if queue[0] == maze.start:
            cur_path = [queue.pop(0)]
        else:
            cur_path = queue.pop(0)
        row,col = cur_path[-1]

        if(row,col) in visited:
            continue
        visited.add((row,col))

        if maze[row,col]==maze.legend.waypoint:
            return cur_path

        for i in maze.neighbors(row,col):
            if i not in visited:
                queue.append(cur_path+[i])
    
    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    visited = set()
    pqueue = PriorityQueue()
    begin = maze.start
    end = maze.waypoints[0]
    heuristic = abs(begin[0] - end[0]) + abs(begin[1] - end[1])
    pqueue.put((heuristic , [begin]))
    while pqueue:
        path = pqueue.get()[1]
        cur_row, cur_col = path[-1]
        for i in maze.neighbors(cur_row, cur_col):
            if maze[i[0], i[1]] == maze.legend.waypoint:
                visited.add(i)
                path += [i]
                return path
                break
            if i not in visited:
                visited.add(i)
                heuristic = abs(i[0] - end[0]) + abs(i[1] - end[1])
                pqueue.put((heuristic+len(path), path+[i]))


    return []
    
def get_MST_length(waypoints_left, maze, edge_weights):
    
    cost = 0
    waypoints_found = []

    current = waypoints_left[0]
    Max = maze.size.x + maze.size.y

    while len(waypoints_left) != 0:

        waypoints_left = list(waypoints_left)
        waypoints_left.remove(current)
        waypoints_left = tuple(waypoints_left)

        waypoints_found.append(current)
        path_length = Max

        for i in waypoints_found:

            for j in waypoints_left:

                if edge_weights[(i,j)] < path_length:
                    path_length = edge_weights[(i,j)]
                    current = j

        cost += path_length  
        
    return cost

def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    begin = maze.start
    farthest_dist = 0
    wp = maze.waypoints
    queue = list(wp)
    #find the farthest distance and farthest waypoint
    for point in queue:
        h = abs(begin[0] - point[0]) + abs(begin[1] - point[1])
        if  (farthest_dist<= h) :
            farthest_dist = h
            farthest_pt = point
    queue.remove(farthest_pt)
    stk = []
    stk.append(farthest_pt)
    current_pt = farthest_pt
    path = []
    while queue:
        #find shortest distance and shortest point
        shortest_dist=1000000
        for i in queue:
            heuristic1=abs(current_pt[0] - i[0]) + abs(current_pt[1] - i[1])
            if (shortest_dist>= heuristic1):
                shortest_dist=heuristic1
                shortest_pt=i
        
        stk.append(shortest_pt)
        current_pt = shortest_pt
        queue.remove(shortest_pt)
        print(shortest_pt)
    stk.append(maze.start)
    while stk:
        temp_start = stk.pop()
        if len(stk) <1:
            break
        temp_end = stk[-1]

        visited=set()
        pqueue=PriorityQueue()
        heuristic2=abs(temp_start[0] - temp_end[0]) + abs(temp_start[1] - temp_end[1])
        pqueue.put((heuristic2,[temp_start]))
        print('second loop')
        #flag represents whether the path is added or not
        flag=True
        while pqueue and flag==True:
            temp_path=pqueue.get()[1]
            tempr,tempc=temp_path[-1]
            for pt in maze.neighbors(tempr,tempc):
                if(flag==False):
                    break
                if (pt[0],pt[1])==temp_end:
                    visited.add(pt)
                    temp_path += [pt]
                    path+=temp_path
                    flag=False
                    #at this point we are free to exit for loop and while loop

                if pt not in visited:
                    visited.add(pt)
                    heuristic3=abs(pt[0] - temp_end[0]) + abs(pt[1] - temp_end[1])
                    g = len(temp_path)
                    pqueue.put((heuristic3+g,temp_path+[pt]))    
                        

        path.pop()
        print('done')
    path += [farthest_pt]
    
    return path



    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
