#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
# Based on: https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
#           by Arthur Juliani
#


import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import ttt_board as ttt
import os.path

tf.reset_default_graph()

model_name = 'tic-tac-toe-model'

def build_graph(sess):
    global inputs1, predict, Qout, updateModel, W, nextQ, saver

    #These lines establish the feed-forward part of the network used to choose actions
    inputs1 = tf.placeholder(shape=[1,9],dtype=tf.float32, name='inputs1')
    W = tf.Variable(tf.random_uniform([9,9],0,0.01), name='W')
    Qout = tf.matmul(inputs1,W, name='Qout')
    predict = tf.argmax(Qout,1, name='predict')

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1,9],dtype=tf.float32, name='nextQ')
    loss = tf.reduce_sum(tf.square(nextQ - Qout), name='loss')
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1, name='trainer')
    updateModel = trainer.minimize(loss, name='updateModel')
    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver()

def load_graph(sess):
    global inputs1, predict, Qout, updateModel, W, nextQ, saver

    saver = tf.train.import_meta_graph(model_name+'.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    all_vars = tf.get_collection('vars')

    graph = tf.get_default_graph()
    inputs1 = graph.get_tensor_by_name("inputs1:0")
    W = graph.get_tensor_by_name("W:0")
    Qout = graph.get_tensor_by_name("Qout:0")
    predict = graph.get_tensor_by_name("predict:0")
    nextQ = graph.get_tensor_by_name("nextQ:0")
    updateModel = graph.get_operation_by_name("updateModel")

# Training the network

# Set learning parameters
y = .99
e = 0.1
num_episodes = 20000000
#create lists to contain total rewards and steps per episode
jList = []
rList = []

board = ttt.Board()

def find_best_legal_move(board, q):
    bestPos = -1
    for i in range(9):
        if (board.is_legal(i) and (bestPos == -1 or q[0,i] > q[0, bestPos])):
            bestPos = i

    return bestPos

with tf.Session() as sess:
    wins=0
    loss=0
    draws=0
    if os.path.exists(model_name+'.meta'):
        load_graph(sess)
    else:
        build_graph(sess)

    for i in range(num_episodes):
        #Reset environment and get first new observation
        board.reset()
        d = False
        #The Q-Network
        while not d:
            #Choose an action by greedily (with e chance of random action) from the Q-network
            feed_dict = {inputs1: [board.state]}
            a,allQ = sess.run([predict,Qout],feed_dict)
            if not board.is_legal(a[0]):
                a[0] = find_best_legal_move(board, allQ)


            if np.random.rand(1) < e:
                a[0] = board.random_empty_spot()

            #Make neural net move and get new state and reward from environment
            old_state = board.state
            _,r,d = board.move(a[0], ttt.NAUGHT)

            #If game not over make a radom opponent move
            if not d:
                s1, r, d = board.move(board.random_empty_spot(), ttt.CROSS)
                r = -r

            reward = 0
            if d:
                if r == ttt.NEUTRAL:
                    reward = 0.5
                elif r == ttt.WIN:
                    reward = 1
                else:
                    reward = 0
                targetQ = allQ
                targetQ[0, a[0]] = reward + y*targetQ[0, a[0]]
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1: [old_state], nextQ: targetQ})
            else:
                #Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout,feed_dict={inputs1: [s1]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = reward + y*maxQ1
                #Train our network using target and predicted Q values
                _,W1 = sess.run([updateModel,W],feed_dict={inputs1: [old_state],nextQ:targetQ})

            if d == True:

                if r > 0:
                    wins = wins+1
#                    print 'WIN'
                elif r < 0:
                    loss =loss+1
#                    print 'LOSS'
                else:
                    draws = draws+1
#                   print 'DRAW'

                e = 1. / ((i / 50) + 10)

                if(i % 100 == 0):
#                    board.print_board()
                    print 'wins: {} Losses: {} Draws: {}'.format(wins, loss, draws)
                    if(loss>0):
                        print 'Ratio:{}'.format(wins*1.0/loss)
                    #Reduce chance of random action as we train the model.
                    if(i%10000):
                        saver.save(sess, model_name)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"

print("End")