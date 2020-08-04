# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:14:37 2019

@author: HP
"""
# -*- coding: utf-8 -*-
import numpy as np
iterations = 5000
lr = 0.5
HiddenNodes = 20
OutputNodes = 1

class NNBike(object):
    def __init__(self, InputNodes, HiddenNodes, OutputNodes, lr):
       	
        # Number of Input,Hidden and Output nodes 
        self.InputNodes = InputNodes
        self.HiddenNodes = HiddenNodes
        self.OutputNodes = OutputNodes
	    #Randomly intializing weights
        self.Input_Weights = np.random.normal(0.0, self.InputNodes**-0.5,(self.InputNodes, self.HiddenNodes))
        self.Hidden_Weights = np.random.normal(0.0, self.HiddenNodes**-0.5,(self.HiddenNodes, self.OutputNodes))
        self.lr = lr
        #Sigmoid function activation
        self.activation_function = lambda x : 1/(1 + np.exp(-x)) 
    
    def train(self, features, targets):
        
        #Train network to implement forwardpass and backpropagation
        noofrecords  = features.shape[0]
        delta_weights_i_h = np.zeros(self.Input_Weights.shape)
        delta_weights_h_o = np.zeros(self.Hidden_Weights.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forwardpass(X)  
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,delta_weights_i_h, delta_weights_h_o)
        self.Weightupdation(delta_weights_i_h, delta_weights_h_o, noofrecords )


    def forwardpass(self, X):
        
        hidden_inputs = np.dot(X, self.Input_Weights) 
        hidden_outputs = self.activation_function(hidden_inputs) 
        final_inputs = np.dot(hidden_outputs, self.Hidden_Weights)
        final_outputs = final_inputs
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        
        # Based on error value of weights is changed here
        error = y - final_outputs 
        hidden_error = np.dot(self.Hidden_Weights, error) 
        output_error_term = error
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        delta_weights_i_h += hidden_error_term * X[:,None]
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        return delta_weights_i_h, delta_weights_h_o

    def Weightupdation(self, delta_weights_i_h, delta_weights_h_o, noofrecords ):
     
        self.Hidden_Weights += self.lr*delta_weights_h_o/noofrecords 
        self.Input_Weights += self.lr*delta_weights_i_h/noofrecords 

    def run(self, features):
      
        hidden_inputs =  np.dot(features, self.Input_Weights)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.Hidden_Weights)
        final_outputs = final_inputs 
        
        return final_outputs



