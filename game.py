import numpy as np
import copy

class connect4:
    def __init__(self, width = 7, height = 6, player1 = 1, player2 = 0.5):
        self.height = height
        self.width = width
        self.players = [player1, player2]
        self.last_move = 0
        self.board = np.zeros((height, width))
        self.Done = False

    def check_win(self, player, do_print = False):
        # check horizontal spaces
        for y in range(self.height):
            for x in range(self.width - 3):
                if self.board[y, x] == self.players[player] and self.board[y, x + 1] == self.players[player] and self.board[y, x + 2] == self.players[player] and self.board[y, x + 3] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('horizontal')
                    self.Done = True
                    return True

        # check vertical spaces
        for y in range(self.height - 3):
            for x in range(self.width):
                if self.board[y, x] == self.players[player] and self.board[y + 1, x] == self.players[player] and self.board[y + 2, x] == self.players[player] and self.board[y + 3, x] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('vertical')
                    self.Done = True
                    return True

        # check \ diagonal spaces
        for y in range(self.height - 3):
            for x in range(self.width - 3):
                if self.board[y, x] == self.players[player] and self.board[y + 1, x + 1] == self.players[player] and self.board[y + 2, x + 2] == self.players[player] and self.board[y + 3, x + 3] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('\ diag')
                    self.Done = True
                    return True

        # check / diagonal spaces
        for y in range(self.height - 3):
            for x in range(self.width - 3):
                if self.board[y + 3, x] == self.players[player] and self.board[y + 2, x + 1] == self.players[player] and self.board[y + 1, x + 2] == self.players[player] and self.board[y, x + 3] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('/ diag', x, y)
                    self.Done = True
                    return True
        return False

    def print_win(self, player):
            if player == self.players[0]:
                print('Player X wins!')
            else:
                print('Player O wins!')

    def print_board(self):
        rows = ['F', 'E', 'D', 'C', 'B', 'A']
        for i in range(self.board.shape[0]):
            print(rows[i] + ' | ', end='')
            for j in range(self.board.shape[1]):
                if (self.board[i, j]) == 0:
                    print(' ', '|', end=' ')
                elif (self.board[i, j]) == self.players[0]:
                    print('X', '|', end=' ')
                elif (self.board[i, j]) == self.players[1]:
                    print('O', '|', end=' ')
            print('')
        print('  | 0 | 1 | 2 | 3 | 4 | 5 | 6 |')

    def place(self, move, player):
        self.board_before = copy.copy(self.board)
        if not self.check_valid(move):
            return False
        for i in range(self.height):
            if (self.board[(self.height - i - 1), move] == 0):
                self.board[(self.height - i - 1), move] = self.players[player]
                return True

    def check_valid(self, move):
        if move > self.width-1:
            return False
        if move < 0:
            return False
        if self.board[0, move]:
            return False
        return True

    def check_draw(self):
        for i in self.board[0, :]:
            if not i:
                return False
        else:
            self.Done = True
            return True

    def human_move(self, player,  return_move=True):
        while(1):
            if player ==0:
                print('Player X to move, pick a row')
            else:
                print('Player O to move, pick a row')
            move = int(input())
            valid = self.check_valid(move)
            if not valid:
                print('Invalid move')
            else:
                break
        if return_move:
            return move