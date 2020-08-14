import numpy as np
import os
import math
import random
import json
from tkinter import *
import tkinter.ttk as ttk


def get_hash(board: np.array):
    """method that returns the board in the shape (9, 0) as a string e.g. '1'"""
    return str(board.reshape(9))


class Agent:
    """Agent class that imitates a human player

    uses Reinforcement Q-Learning to get better at the game over time

    low learning rate and high discount factor are recommended as the game is rather short

    gets high positive reward for winning
    low positive reward for tie and
    negative reward for losing

    for having a good opponent simulating at least 20000 games is recommended"""

    def __init__(self, epsilon: float, alpha: float, gamma: float, index: int, symbol: str, save_file_path=None):
        """constructor of the Player class

        params:
        epsilon (float) - exploration rate (probability of the agent performing a random move)
        alpha (float) - learning rate (how much the new value changes the old value)
        gamma (float) - discount factor (importance for future rewards)
        index (int) - the player's index
        symbol (int) - the player's symbol
        save_file_path (str) - (optional) path to a json file containing q_values

        attributes:
        q_values (dict) - q_values
        states (list) - all states of one game"""
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.index = index
        self.symbol = symbol
        self.q_values = dict()
        self.states = list()

        if save_file_path is not None:
            self.load_q_values(save_file_path)

    def save_q_values(self, path):
        """saves all q_values as json at the given path"""
        if path[-5:] != '.json':
            path += '.json'
        with open(path, 'w') as save_file:
            json.dump(self.q_values, save_file)

    def load_q_values(self, path):
        """loads all q_values from the given json file and overrides own q_values with the new ones"""
        if path[-5:] != '.json':
            path += '.json'
        with open(path) as save_file:
            self.q_values = json.load(save_file)

    def choose_action(self, positions: list, current_board: np.array, exploration=True):
        """method to choose an action

        args:
        positions (list) - all available positions on the board
        current_board (np.array) - the current state of the board

        return:
        action (tuple) - the row and column of the field that shall be changed"""

        if exploration and np.random.uniform(0, 1) <= self.epsilon:
            return Agent.choose_random_action(positions)

        max_value = -math.inf
        action = tuple()

        for position in positions:
            next_board = current_board.copy()
            next_board[position] = self.index
            next_board_hash = get_hash(next_board)
            value = 0 if self.q_values.get(next_board_hash) is None else self.q_values.get(next_board_hash)

            if value > max_value:
                max_value = value
                action = position

        next_board = current_board.copy()
        next_board[action[0], action[1]] = self.index
        self.states.append(get_hash(next_board))
        return action

    @staticmethod
    def choose_random_action(positions: list):
        """method to let the agent make a random action"""
        return random.choice(positions)

    def feed_reward(self, reward):
        for state in self.states:
            if self.q_values.get(state) is None:
                self.q_values[state] = 0
                continue
            self.q_values[state] = (1 - self.alpha) * self.q_values[state] + self.alpha * (reward + self.gamma)

    def __str__(self):
        return

    def __repr__(self):
        return self.__doc__


class Game:
    """class that controls the tic-tac-toe game"""

    def __init__(self):
        """constructor of the Game class

        params:
        buttons (list) - list of all buttons of the TicTacToe GUI
        attributes:

        board: np.array() - multidimensional np array with three rows and columns containing the board
        player_turn: int - the index of the player who will make the next move (1 or 2)
        symbols_placed
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.player_turn = 1  # attribute that indicates which player may proceed (1 or 2)
        self.symbols_placed = 0
        self.total_games = 0  # count of all played games
        self.buttons = []
        self.win_label = None
        self.player1 = Agent(0.3, 0.2, 0.9, 1, 'X')
        self.player2 = Agent(0.2, 0.3, 0.9, 2, 'O')

    def append_buttons(self, button: Button):
        self.buttons.append(button)

    def train(self, iterations: int, master: Tk, progress_bar: ttk.Progressbar, label):
        """method that trains both players

        params:
        iterations (int) - the number of games the two agents will play
        master (Tk) - tkinter master/root
        progress_bar (tkinter.tkk.Progressbar) - progressbar on GUI showing loading state"""
        for i in range(iterations):
            if (i + 1) % 50 == 0:
                master.update()
                progress_bar['value'] = (i / iterations) * 100
                label['text'] = f'{i + 1}/{iterations} games simulated'

            while True:
                player_action = tuple()
                if self.player_turn == 1:
                    player_action = self.player1.choose_action(self.get_available_positions(), self.board)
                elif self.player_turn == 2:
                    player_action = self.player2.choose_action(self.get_available_positions(), self.board)
                self.change_field(player_action)

                if self.check_win(training=True) != 0:
                    self.reset()
                    break
        progress_bar['value'] = 100
        label['text'] = f'successfully simulated {iterations} games.'
        tk.update()

    def change_field(self, indices: tuple, update_gui=False):
        """method to change a field in the game's board

        params:
        index_row: int - the number of the row of the field that shall be changed
        index_col: int - the number of the column of the field that shall be changed
        """
        if not self.board[indices] == 0:
            return

        self.symbols_placed += 1
        self.board[indices] = self.player1.index if self.player_turn == 1 else self.player2.index

        if update_gui:
            symbol = self.player1.symbol if self.player_turn == 1 else self.player2.symbol
            self.buttons[indices[0] * 3 + indices[1]]['text'] = symbol  # changes the pressed button's label
        self.player_turn = abs(self.player_turn - 3)  # switches player_turn between 1 and 2

    def check_win(self, training: bool = True, disable_buttons: bool = False):
        """method that determines if one of the two players has won the game

        return:
        0 - no one has won
        1 - player1 has won
        2 - player2 has won
        3 - it's a tie"""
        win_index = 0

        # horizontal win
        for row in self.board:
            if np.count_nonzero(row == row[0]) == len(row) and row[0] != 0:
                win_index = row[0]

        # vertical win
        for col in range(self.board.shape[1]):
            check = list()

            for row in self.board:
                list(row)
                check.append(row[col])

            if check.count(check[0]) == len(check) and check[0] != 0:
                win_index = check[0]

        # diagonal win
        diagonals = list()
        for index in range(self.board.shape[0]):
            diagonals.append(self.board[index, index])
        if diagonals.count(diagonals[0]) == len(diagonals) and diagonals[0] != 0:
            win_index = diagonals[0]

        diagonals = list()
        for index_row, index_col in zip(range(self.board.shape[0]), range(self.board.shape[0] - 1, -1, -1)):
            diagonals.append(self.board[index_row, index_col])
        if diagonals.count(diagonals[0]) == len(diagonals) and diagonals[0] != 0:
            win_index = diagonals[0]

        # tie
        if self.symbols_placed >= len(self.board) ** 2 and win_index == 0:
            win_index = 3

        if win_index != 0:
            if training:
                self.calculate_rewards(win_index)
            if disable_buttons:
                self.disable_buttons()

        return win_index

    def calculate_rewards(self, win_index, win_reward=10, lose_reward=-10, tie_reward=2):
        """feeds rewards to both players depending on how the game ended

        params:
        win_index (int) - return of self.check_game"""
        if win_index == 0:
            return
        if win_index == 1:
            self.player1.feed_reward(win_reward)
            self.player2.feed_reward(lose_reward)
        elif win_index == 2:
            self.player1.feed_reward(lose_reward)
            self.player2.feed_reward(win_reward)
        elif win_index == 3:
            self.player1.feed_reward(tie_reward)
            self.player2.feed_reward(tie_reward)

    def reset(self, update_gui: bool = False):
        """method to reset the whole game

        params:
        update_gui (bool) - when True the GUI also gets reset and all buttons are empty again"""
        self.board = np.zeros((3, 3), dtype=int)
        self.player_turn = 1
        self.symbols_placed = 0
        self.total_games += 1
        self.player1.states = list()
        self.player2.states = list()
        if update_gui:
            self.win_label['text'] = ''
            for button in self.buttons:
                button['state'] = NORMAL
                button['text'] = ''

    def reset_ai(self, label):
        self.reset(update_gui=True)
        self.player1.q_values = dict()
        self.player2.q_values = dict()

        if label is not None:
            label['text'] = 'AI successfully reset'

    def get_available_positions(self):
        """iterates over the board and returns all available fields

        return:
        available_fields (list) - list of tuples each containing row and column of a free field"""
        available_fields = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    available_fields.append((i, j))
        return available_fields

    def player_win(self, player_index: int):
        """method that outputs which player has won

        params:
        player_index (int) - (1, 2 or 3) the index of the player who won (should come from Game.check_win() output)"""
        message = ''
        if player_index == 1:
            message = 'Player1 wins'
        elif player_index == 2:
            message = 'Player2 wins'
        elif player_index == 3:
            message = 'it\'s a tie'

        if self.win_label is not None:
            self.win_label['text'] = message
            self.disable_buttons()

    def disable_buttons(self):
        for button in self.buttons:
            button['state'] = DISABLED

    def __str__(self):
        return self.board

    def __repr__(self):
        return f'{self.__doc__} \n{self.board} \n' \
               f'<{type(self).__name__}.symbols_placed: {self.symbols_placed}, ' \
               f'{type(self).__name__}.player_turn: {self.player_turn}, ' \
               f'{type(self).__name__}.check_win: {self.check_win()}>'


def get_win():
    win = game.check_win()
    if win != 0:
        game.player_win(win)
        game.reset()
        return True


def create_button(command, master):
    return Button(master, text='', font=('Helvetica', 20, 'bold'), bg='#c9c9c9', fg='black',
                  height=4, width=8, command=command)


def load_game():
    try:
        training_count_entry.config(state=DISABLED)
        start_button.config(state=DISABLED)

        game.train(int(training_count_entry.get()), tk, progress, info_label)

        training_count_entry.config(state=NORMAL)
        start_button.config(state=NORMAL)
    except TclError:
        tk.quit()
        os._exit(0)
    except ValueError:
        info_label['text'] = 'You have to enter an integer!'
    finally:
        training_count_entry.config(state=NORMAL)
        start_button.config(state=NORMAL)


if __name__ == '__main__':

    """training and loading progress"""

    tk = Tk()
    tk.config(background='#fff')
    tk.resizable(0, 0)  # window can not be resized

    style = ttk.Style()
    style.configure('tab.TFrame', background='#fff')

    tab_parent = ttk.Notebook(tk)
    tab_parent.pack()

    training_tab = ttk.Frame(tab_parent)  # training tab (tab 1)
    training_tab.config(style='tab.TFrame')
    tab_parent.add(training_tab, text='Training')

    game = Game()

    game_label = Label(training_tab, text='Tic Tac Toe', font=('Helvetica', 35, 'bold'), bg='#fff', fg='#000')
    game_label.grid(row=0, column=0, columnspan=3, pady=20)

    training_count_label = Label(training_tab, text='training count', font=('Helvetica', 15), bg='#fff', fg='#000')
    training_count_label.grid(row=1, column=0)
    training_count_entry = Entry(training_tab, bd=4)
    training_count_entry.grid(row=1, column=1)

    start_button = Button(training_tab, text='Start', font=('Helvetica', 15, 'bold'), bg='#6c9c62', fg='#000',
                          height=1, width=5, command=load_game)
    start_button.grid(row=1, column=2)

    progress = ttk.Progressbar(training_tab, orient=HORIZONTAL, length=400, mode='determinate')
    progress.grid(row=2, column=0, columnspan=3, padx=20, pady=20)

    info_label = Label(training_tab, text='', font=('Helvetica', 15, 'bold'), bg='#fff', fg='#000')
    info_label.grid(row=3, column=0, columnspan=3)

    reset_ai_button = Button(training_tab, text='RESET AI', font=('Helvetica', 15, 'bold'), bg='#8c3838', fg='#000',
                             height=2, width=8, command=lambda: game.reset_ai(info_label))
    reset_ai_button.grid(row=4, column=0, columnspan=3, padx=20, pady=30)

    """game progress"""

    game_tab = ttk.Frame(tab_parent)  # training tab (tab 1)
    game_tab.config(style='tab.TFrame')
    tab_parent.add(game_tab, text='Game')

    game_label = Label(game_tab, text='Tic Tac Toe', font=('Helvetica', 35, 'bold'), bg='#fff', fg='#000')
    game_label.grid(row=0, column=0, columnspan=3)

    button0 = create_button(lambda: game.change_field((0, 0), update_gui=True), game_tab)
    button0.grid(row=1, column=0)
    game.buttons.append(button0)

    button1 = create_button(lambda: game.change_field((0, 1), update_gui=True), game_tab)
    button1.grid(row=1, column=1)
    game.buttons.append(button1)

    button2 = create_button(lambda: game.change_field((0, 2), update_gui=True), game_tab)
    button2.grid(row=1, column=2)
    game.buttons.append(button2)

    button3 = create_button(lambda: game.change_field((1, 0), update_gui=True), game_tab)
    button3.grid(row=2, column=0)
    game.buttons.append(button3)

    button4 = create_button(lambda: game.change_field((1, 1), update_gui=True), game_tab)
    button4.grid(row=2, column=1)
    game.buttons.append(button4)

    button5 = create_button(lambda: game.change_field((1, 2), update_gui=True), game_tab)
    button5.grid(row=2, column=2)
    game.buttons.append(button5)

    button6 = create_button(lambda: game.change_field((2, 0), update_gui=True), game_tab)
    button6.grid(row=3, column=0)
    game.buttons.append(button6)

    button7 = create_button(lambda: game.change_field((2, 1), update_gui=True), game_tab)
    button7.grid(row=3, column=1)
    game.buttons.append(button7)

    button8 = create_button(lambda: game.change_field((2, 2), update_gui=True), game_tab)
    button8.grid(row=3, column=2)
    game.buttons.append(button8)

    win_label = Label(game_tab, text='', font=('Helvetica', 20, 'bold'), bg='#fff')
    win_label.grid(row=4, column=0, columnspan=2)
    game.win_label = win_label

    reset_button = Button(game_tab, text='reset', font=('Helvetica', 20, 'bold'), bg='#1c1c1c', fg='white',
                          height=2, width=8, command=lambda: game.reset(update_gui=True))
    reset_button.grid(row=4, column=2)

    # game.player1.save_q_values('player1')
    # game.player2.save_q_values('player2')

    # mainloop
    while True:
        while True:
            try:
                if not game.player_turn == 1:
                    if get_win():
                        break
                    agent_action = game.player2.choose_action(game.get_available_positions(),
                                                              game.board, exploration=False)
                    game.change_field(agent_action, update_gui=True)
                    if get_win():
                        break

                tk.update()
                tk.update_idletasks()
            except TclError:
                tk.quit()
                sys.exit()
