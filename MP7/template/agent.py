import numpy as np
import math
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.grid_size = utils.GRID_SIZE
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        Sprime= self.generate_state(environment)
        if self.a is not None and self._train and self.s is not None:
            reward = -0.1
            if points > self.points:
                reward = 1
            elif dead:
                reward = -1

            self.N[self.s][self.a] +=1
            learning_rate=self.C / (self.C + self.N[self.s][self.a])
            maxQ = -math.inf
            for i in self.actions:
                if  self.Q[Sprime][i] >= maxQ:
                    maxQ = self.Q[Sprime][i]
            Qprime=self.Q[self.s][self.a] + learning_rate* (reward+self.gamma*maxQ -self.Q[self.s][self.a])
            self.Q[self.s][self.a] = Qprime

        if dead :
            self.reset()
            return 0
        else:
            self.s = Sprime
            self.points = points
        best_action=0
        maxf = -math.inf
        #find the optimal action with maxf
        for j in self.actions:
            n_table = self.N[Sprime][j]
            if self.Ne > n_table:
                val=1
            else:
                val=self.Q[Sprime][j]
            if maxf<=val:
                maxf = val
                best_action = j

        ##self.N[Sprime][best_action] += 1
        self.a = best_action
        return self.a
        


    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        ''''''
        (head_x, head_y, snake_body, food_x, food_y) = environment
        grid=self.grid_size
        #check food direction
        food_xdirection = 2
        if head_x == food_x:
            food_xdirection = 0
        else:
            if head_x > food_x:
                food_xdirection = 1

        food_ydirection = 2
        if head_y == food_y:
            food_ydirection = 0
        else:
            if head_y > food_y:
                food_ydirection = 1
    #check adjoining wall
        awall_x = 0
        if head_x == grid:
            awall_x = 1
        else:
            if head_x == (utils.DISPLAY_SIZE/grid -2) * grid:
                awall_x = 2
        awall_y= 0
        if  head_y == grid:
            awall_y=1
        else:
            if head_y == (utils.DISPLAY_SIZE/grid -2) * grid:
                awall_y= 2


        #check if grid in snake body
        adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = 0, 0, 0, 0
        for (x, y) in snake_body:
            if head_x == x: 
                if y == head_y - grid:
                    adjoining_body_top = 1
                elif head_y + grid:
                    adjoining_body_bottom = 1
            if head_y == y:
                if x == head_x - grid: 
                    adjoining_body_left = 1
                elif x == head_x + grid:
                    adjoining_body_right = 1

        return (food_xdirection, food_ydirection, awall_x, awall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        