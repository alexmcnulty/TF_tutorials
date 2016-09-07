# Uploaded as a .py file rather than an ipython notebook as I didn't want to endure the struggle of getting the animation to display inline in the notebook

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

import matplotlib.animation as animation
from IPython.display import HTML

# Google 'Conway's Game of Life' for details


# To generate a starting board we simply need a matrix of 0's and 1's
# and then print this as black and white



# Credit for updating idea: Jake VanderPlas
# Use the scipy function convolve2d to count # of neighbours
# Then apply the runs of the game - a cell is alive if it has 3 neighbours
# or if it was already alive and has 2 neighbours, otherwise it is dead

shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)


from scipy.signal import convolve2d
def update_board(X):
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    X = (N == 3) | (X & (N == 2))
    return X

# convert function to a Tf node using py_func

board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])
fig = plt.figure()

# Use the code given to run 100 iterations of the game and produce an animation
count = 0
with tf.Session() as session:
    X = session.run(initial_board)
    plot = plt.imshow(X, cmap='Greys', interpolation='nearest')

    def game_of_life(*args):
        global X, count
        if count < 100:
            X = session.run(board_update, feed_dict={board: X})[0]
            count += 1
        plot.set_array(X)
        return plot,

    ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=False)
    plt.show()
