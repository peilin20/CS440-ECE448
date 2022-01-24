# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    center=alien.get_centroid()
    wid=alien.get_width()
    len=alien.get_length()
    ht=alien.get_head_and_tail()
    head=ht[0]
    tail=ht[1]

    for wall in walls:
        if alien.is_circle():
            check=does_circle_touch(wall,granularity,center,wid)
            if check:
                return check
        else:
            if np.isclose(head[1],tail[1]):
                x=min(head[0],tail[0])
                while max(head[0],tail[0])>=x:
                    check=does_circle_touch(wall,granularity,(x,head[1]),wid)
                    if check:
                        return check
                    x=x+1
            if np.isclose(head[0],tail[0]):
                y=min(head[1],tail[1])
                while max(head[1],tail[1])>=y:
                    check=does_circle_touch(wall,granularity,(head[0],y),wid)
                    if check:
                        return check
                    y=y+1

    return False

class pt:
    def __init__(self, x, y):
        self.x=x
        self.y=y

def does_circle_touch(wall,granularity, centorid, radius):
    g=granularity/(math.sqrt(2))
    top=g+radius
    
    pt1x=centorid[0]-wall[0]
    pt1y=centorid[1]-wall[1]
    length1=math.sqrt(pt1x**2 + pt1y**2)

    pt2x=wall[2]-wall[0]
    pt2y=wall[3]-wall[1]
    length2=math.sqrt(pt2x**2 + pt2y**2)

    if length2 <1 and length1-top<=0:
        return True
    projection= (pt1x*pt2x +pt1y*pt2y)/length2
    distance= abs((pt1x*pt2y - pt1y*pt2x)/length2)
    if projection>=0: 
        if length2>=projection and distance-top<=0:
            return True
    elif length2 <projection:
        pt3x=pt1x-pt2x
        pt3y=pt1y-pt2y
        length3=math.sqrt(pt3x**2+pt3y**2)
        if length3-top<=0:
            return True
    else:
        if length1-top<=0:
            return True
        return False
def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    wid=alien.get_width()
    center=alien.get_centroid()
    ht=alien.get_head_and_tail()
    for pt in goals:
        if alien.is_circle():
            bound=math.sqrt(abs(center[0]-pt[0])**2 + abs(center[1]-pt[1])**2)
            if 0>= bound-wid-pt[2]:
                return True
            
        else:
            radius=pt[2]+wid
            w=(ht[0][0],ht[0][1],ht[1][0],ht[1][1])
            check=does_circle_touch(w,0,(pt[0],pt[1]),radius)
            if check:
                return check
    return False

def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    center=alien.get_centroid()
    ht=alien.get_head_and_tail()
    wid=window[0]
    len=window[1]
    if center[0] <=0 or ht[0][0] <=0 or ht[1][0] <=0:
        return False
    if window[0]<=center[0] or window[0]<=ht[0][1] or window[0]<=ht[1][1]:
        return False
    if window[1]<=center[1] or window[1]<=ht[0][1] or window[1]<=ht[1][1]:
        return False
    wall=[(0,0,wid,0),(0,0,0,len),(wid,0,wid,len),(0,len,wid,len)]
    check=does_alien_touch_wall(alien,wall,granularity)
    if not check:
        return True
    else:
        return False

if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0) 
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, expected: {truths[0]}'
        assert touch_goal_result == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, expected: {truths[1]}'
        assert in_window_result == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, expected: {truths[2]}'

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")