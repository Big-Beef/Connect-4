import numpy as np
class connect4:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.board = np.zeros((height, width))

    def check_win(self, player):
        # check horizontal spaces
        for y in range(self.height):
            for x in range(self.width - 3):
                if self.board[y, x] == player and self.board[y, x + 1] == player and self.board[y, x + 2] == player and self.board[y, x + 3] == player:
                    # self.print_win(player)
                    # print('horizontal')
                    return True

        # check vertical spaces
        for y in range(self.height - 3):
            for x in range(self.width):
                if self.board[y, x] == player and self.board[y + 1, x] == player and self.board[y + 2, x] == player and self.board[y + 3, x] == player:
                    # self.print_win(player)
                    # print('vertical')
                    return True

        # check \ diagonal spaces
        for y in range(self.height - 3):
            for x in range(self.width - 3):
                if self.board[y, x] == player and self.board[y + 1, x + 1] == player and self.board[y + 2, x + 2] == player and self.board[y + 3, x + 3] == player:
                    # self.print_win(player)
                    # print('\ diag')
                    return True

        # check / diagonal spaces
        for y in range(self.height - 3):
            for x in range(self.width - 3):
                if self.board[y + 3, x] == player and self.board[y + 2, x + 1] == player and self.board[y + 1, x + 2] == player and self.board[y, x + 3] == player:
                    # self.print_win(player)
                    # print('/ diag', x, y)
                    return True
        return False

    def print_win(self, player):
            if player == 1:
                print('Player X wins!')
            else:
                print('Player O wins!')

    def check_valid(self, move):
        if move > self.width-1:
            return False
        if move < 0:
            return False
        if self.board[0, move]:
            return False
        else:
            return True

    def check_draw(self):
        for i in self.board[0, :]:
            if not i:
                return False
        else:
            return True

    def place(self, move, player):
        move = move
        if not self.check_valid(move):
            return False

        else:
            for i in range(self.height):
                if (self.board[(self.height-i-1), move] == 0):
                    self.board[(self.height-i-1), move] = player
                    return True

    def move(self, player):
        while(1):
            if player == 1:
                print('Player X to move, pick a row')
            else:
                print('Player O to move, pick a row')
            move = int(input())
            valid = self.place(move, player)
            if not valid:
                print('Invalid move')
            else:
                break

    def print_board(self):
        rows = ['F', 'E', 'D', 'C', 'B', 'A']
        for i in range(self.board.shape[0]):
            print(rows[i] + ' | ', end='')
            for j in range(self.board.shape[1]):
                if int(self.board[i, j]) == 0:
                    print(' ', '|', end=' ')
                elif int(self.board[i, j]) == 1:
                    print('X', '|', end=' ')
                elif int(self.board[i, j]) == -1:
                    print('O', '|', end=' ')
            print('')
        print('  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |')