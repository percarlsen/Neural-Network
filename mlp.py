# coding=utf-8
''' Feel free to use numpy for matrix multiplication and
	other neat features.
	You can write some helper functions to
	place some calculations outside the other functions
	if you like to.

	This pre-code is a nice starting point, but you can
	change it to fit your needs.
'''
import numpy as np
import random
import math
import sys

class mlp:
	def __init__(self, inputs, targets, nhidden):
		self.beta = 1
		self.eta = 0.1
		self.momentum = 0.0

                self.noutput = 8
                self.nhidden = nhidden
                self.in_weights_hidden = np.random.uniform(-1,1,size = (len(inputs[0])+1, nhidden))
                self.out_weights_hidden = np.random.uniform(-1,1,size = (nhidden+1, 8))

                self.result = [0]*self.noutput #result array
                self.vhidden = [0]*self.nhidden #hidden layer value

                self.sum_delta = 0
                self.iteration_limit = 10
                self.validation_set_error = sys.maxint

        def earlystopping(self, inputs, targets, valid, validtargets):
                inputs = np.concatenate((inputs,-np.ones((len(inputs),1))),axis=1) #bias
                valid = np.concatenate((valid, -np.ones((len(valid),1))), axis=1) #bias
                
                cnt = 0
                while(True): #loop until overfit
                        delta = 0
                        for i in range (0, self.iteration_limit): #learn!
                                self.train(inputs, targets, len(inputs))
                        
                        for i in range(0, len(valid)): #check if overfit
                                self.forward(valid[i])
                                delta += self.result - validtargets[i]

                        delta = delta/len(valid) #get average
                        delta = sum(np.absolute(delta))
                        if delta > self.validation_set_error: #new average error greater than last
                                if cnt == 2: #stop training
                                        print "Training stopped, optimum reached"
                                        break #stop learning (overfit)
                                else: #x iterations in a row must be worse to stop trainig, to avoid small local minimum
                                        print "Still training"
                                        cnt += 1 
                        else: 
                                cnt = 0
                                print "Still training"
                                self.validation_set_error = delta #all good, update value and continue
                

	def train(self, inputs, targets, iterations):
                for i in range(0,iterations):
                        self.forward(inputs[i])
                        self.backward(targets[i], inputs[i])

                #Shuffle data for next iteration:
                order = list(range(np.shape(inputs)[0]))
                np.random.shuffle(order)
                inputs = inputs[order,:]
                targets = targets[order,:]

	def forward(self, inputs):
                hj =  np.dot(inputs, self.in_weights_hidden)
                self.vhidden = 1/(1+np.exp(-self.beta*hj)) #update hidden nodes value
                
                self.vhidden = np.append(self.vhidden, -1) #add bias
                self.result = np.dot(self.vhidden, self.out_weights_hidden) #update result value

        def backward(self, target, inputs):
                ones = np.array([1]*len(self.vhidden)) #[1,1,1,...,1]
                
                deltao = (self.result-target) #error output layer
                deltah = self.vhidden*(ones-np.array(self.vhidden)) * np.dot(deltao, np.transpose(self.out_weights_hidden)) #error hidden layer

                for i in range(0, len(self.in_weights_hidden)): #update hidden layer weights
                        for j in range(0, self.nhidden):
                                self.in_weights_hidden[i][j] -= self.eta*deltah[j]*inputs[i]

                for i in range(0,len(self.out_weights_hidden)-1): #update output layer weights
                        for j in range(0, self.noutput):
                                self.out_weights_hidden[i][j] -= self.eta*self.vhidden[i]*deltao[j]


	def confusion(self, inputs, targets):
                result_matrix = []
                inputs = np.concatenate((inputs,-np.ones((len(inputs),1))),axis=1) #bias

                for i in range (0,len(inputs)): #get result for each input
                        self.forward(inputs[i])
                        result_matrix.append(self.result)

                result_matrix = np.argmax(result_matrix,1)
                targets = np.argmax(targets,1)
                        
                cm = np.zeros((self.noutput,self.noutput))
                for i in range(self.noutput):
                        for j in range(self.noutput):
                                cm[i,j] = np.sum(np.where(result_matrix==i,1,0)*np.where(targets==j,1,0))
                                        
                print "Confusion matrix:"
                print cm
                print "Percentage correct: ",np.trace(cm)/np.sum(cm)*100, "%"
