import numpy as np
class connect4:
    def __init__(self, width = 7, height = 6, player1 = 1, player2 = -1):
        self.height = height
        self.width = width
        self.players = [player1, player2]
        self.last_move = 0
        self.board = np.zeros((height, width))

    def check_done(self, player, do_print = False, reward = False):
        done = False
        R1 = 0
        R2 = 0
        win = self.check_win(player, do_print)
        draw = self.check_draw()

        if win:
            R1 = 1
            R2 = -1
            done = True
        if draw:
            R1 = -0.1
            R2 = -0.1
            done = True

        if reward:
            return done, R1, R2
        else:
            return done

    def RL_play(self, player, move):
        board_view_before_p1 = self.board.copy().reshape(-1)
        board_view_before_p2 = self.board.copy().reshape(-1) * -1
        if not self.place(move, player):
            return False


        done, R1, R2 = self.check_done(player, reward=True)
        experience1 = {'observation': board_view_before_p1, 'move': move, 'R': R1, 'Done': done}
        experience2 = {'observation': board_view_before_p2, 'move': self.last_move, 'R': R2, 'Done': done}

        self.last_move = move
        return experience1, experience2, done



    def check_win(self, player, do_print = False):
        # check horizontal spaces
        for y in range(self.height):
            for x in range(self.width - 3):
                if self.board[y, x] == self.players[player] and self.board[y, x + 1] == self.players[player] and self.board[y, x + 2] == self.players[player] and self.board[y, x + 3] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('horizontal')
                    return True

        # check vertical spaces
        for y in range(self.height - 3):
            for x in range(self.width):
                if self.board[y, x] == self.players[player] and self.board[y + 1, x] == self.players[player] and self.board[y + 2, x] == self.players[player] and self.board[y + 3, x] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('vertical')
                    return True

        # check \ diagonal spaces
        for y in range(self.height - 3):
            for x in range(self.width - 3):
                if self.board[y, x] == self.players[player] and self.board[y + 1, x + 1] == self.players[player] and self.board[y + 2, x + 2] == self.players[player] and self.board[y + 3, x + 3] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('\ diag')
                    return True

        # check / diagonal spaces
        for y in range(self.height - 3):
            for x in range(self.width - 3):
                if self.board[y + 3, x] == self.players[player] and self.board[y + 2, x + 1] == self.players[player] and self.board[y + 1, x + 2] == self.players[player] and self.board[y, x + 3] == self.players[player]:
                    if do_print:
                        self.print_win(self.players[player])
                        print('/ diag', x, y)
                    return True
        return False

    def print_win(self, player):
            if player == self.players[0]:
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

    def place(self, move, player, do_move=True):
        move = move
        if not self.check_valid(move):
            return False
        if do_move:
            for i in range(self.height):
                if (self.board[(self.height-i-1), move] == 0):
                    self.board[(self.height-i-1), move] = self.players[player]
                    return True
        return True

    def move(self, player,  return_move, do_move):
        while(1):
            if player ==0:
                print('Player X to move, pick a row')
            else:
                print('Player O to move, pick a row')
            move = int(input())
            valid = self.place(move, player, do_move)
            if not valid:
                print('Invalid move')
            else:
                break
        if return_move:
            return move

    def print_board(self):
        rows = ['F', 'E', 'D', 'C', 'B', 'A']
        for i in range(self.board.shape[0]):
            print(rows[i] + ' | ', end='')
            for j in range(self.board.shape[1]):
                if int(self.board[i, j]) == 0:
                    print(' ', '|', end=' ')
                elif int(self.board[i, j]) == self.players[0]:
                    print('X', '|', end=' ')
                elif int(self.board[i, j]) == self.players[1]:
                    print('O', '|', end=' ')
            print('')
        print('  | 0 | 1 | 2 | 3 | 4 | 5 | 6 |')