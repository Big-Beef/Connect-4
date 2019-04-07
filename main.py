import numpy as np
import matplotlib.pyplot as plt
from functions import check_for_win,check_valid,check_draw

    
width = 7
height = 6
player = 1
board = np.zeros((height,width))
winner = False
draw = False


while not winner and not draw:
    
    valid = False
    while valid == False:
        move = int(input())
        valid = check_valid(board, move)
        
    player = player*-1
    
    for j in range(len(board[:,move])-1,-1,-1):
        if board[j,move] == 0:    
            board[j,move] = player;
            break 

    if check_for_win(board , height , width , player):
        winner = player
        print("Winner is:", player)
    
    draw = check_draw(board)    
    if draw:
        print("It's a dtaw")
    
    plt.matshow(board)
    plt.pause(0.05)
    plt.show