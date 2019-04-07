# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:08:17 2019

@author: Michael
"""

def check_for_win(board , height , width , player):
    # check horizontal spaces
    for y in range(height):
        for x in range(width - 3):
            if board[y][x] == player and board[y][x+1] == player and board[y][x+2] == player and board[y][x+3] == player:
                return True

    # check vertical spaces
    for y in range(height-3):
        for x in range(width):
            if board[y][x] == player and board[y+1][x] == player and board[y+2][x] == player and board[y+3][x] == player:
                return True
      
    # check \ diagonal spaces
    for y in range(height-3):
        for x in range(width-3):
            if board[y][x] == player and board[y+1][x+1] == player and board[y+2][x+2] == player and board[y+3][x+3] == player:
                return True

    # check / diagonal spaces
    for y in range(height-3):
        for x in range(width-3):
            if board[y+3][x] == player and board[y+2][x+1] == player and board[y+1][x+2] == player and board[y][x+3] == player:
                return True
    return False

def check_valid(board , move):
    if board[0,move]:
        return False
    else:
        return True
    
def check_draw(board):
    for i in board[0,:]:
        if not i:
            return False
    else:
        return True