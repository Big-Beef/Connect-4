# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:55:08 2019

@author: Eddie
"""
import numpy as np
from check_for_win import *
from board_update import *



#the purpose of this function is to be used to test the fucntions



width = 7
height = 6
board = np.zeros((height,width))







print(check_for_win(board, height , width , 0))