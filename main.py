from modules import *
from tkinter import *

tk = Tk()
tk.title("Tic Tac Toe")
tk.config(background='#fff')
tk.resizable(0, 0)  # window can not be resized

game = Game()


def create_button(command):
    return Button(tk, text='', font=('Helvetica', 20, 'bold'), bg='#c9c9c9', fg='black',
                  height=4, width=8, command=command)


game_label = Label(tk, text='Tic Tac Toe', font=('Helvetica', 35, 'bold'), bg='#fff', fg='black', height=2)
game_label.grid(row=0, column=0, columnspan=3)

button0 = create_button(lambda: game.change_field((0, 0), update_gui=True))
button0.grid(row=1, column=0)
game.buttons.append(button0)

button1 = create_button(lambda: game.change_field((0, 1), update_gui=True))
button1.grid(row=1, column=1)
game.buttons.append(button1)

button2 = create_button(lambda: game.change_field((0, 2), update_gui=True))
button2.grid(row=1, column=2)
game.buttons.append(button2)

button3 = create_button(lambda: game.change_field((1, 0), update_gui=True))
button3.grid(row=2, column=0)
game.buttons.append(button3)

button4 = create_button(lambda: game.change_field((1, 1), update_gui=True))
button4.grid(row=2, column=1)
game.buttons.append(button4)

button5 = create_button(lambda: game.change_field((1, 2), update_gui=True))
button5.grid(row=2, column=2)
game.buttons.append(button5)

button6 = create_button(lambda: game.change_field((2, 0), update_gui=True))
button6.grid(row=3, column=0)
game.buttons.append(button6)

button7 = create_button(lambda: game.change_field((2, 1), update_gui=True))
button7.grid(row=3, column=1)
game.buttons.append(button7)

button8 = create_button(lambda: game.change_field((2, 2), update_gui=True))
button8.grid(row=3, column=2)
game.buttons.append(button8)

win_label = Label(tk, text='', font=('Helvetica', 20, 'bold'), bg='#fff')
win_label.grid(row=4, column=0, columnspan=2)

reset_button = Button(tk, text='reset', font=('Helvetica', 20, 'bold'), bg='#1c1c1c', fg='white', height=2, width=8,
                      command=lambda: game.reset(update_gui=True))
reset_button.grid(row=4, column=2)

game.train(100000, print_progress=True)
game.player1.save_q_values('player1')
game.player2.save_q_values('player2')


def get_win():
    win = game.check_win()
    if win != 0:
        game.player_win(win, win_label)
        game.reset()
        return True


if __name__ == '__main__':
    # mainloop
    while True:
        # print(game.board)
        while True:
            try:
                if not game.player_turn == 1:
                    if get_win():
                        break
                    agent_action = game.player2.choose_action(game.get_available_positions(), game.board, exploration=False)
                    game.change_field(agent_action, update_gui=True)
                    if get_win():
                        break

                tk.update()
                tk.update_idletasks()
            except TclError:
                tk.quit()
                print('quit game!')
                quit()
