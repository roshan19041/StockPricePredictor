
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:13:30 2019

@author: roshanprakash
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
np.random.seed(12)

class DataHelper:
    
    """ Data Helper to process Historical data and to sample batches of training data when required."""
    
    def __init__(self, data_path, train_size=0.8, batch_size=256, input_steps=28, output_steps=1):
        """
        Sets up model data of shape (N,input_steps,1) and corresponding targets of shape (N,T,output_steps)
        
        PARAMETERS
        ----------
        - data_path (str) : the disk path to the data
        - train_size (float, default=0.8) : the proportion of samples to be included in the training data
        - batch_size (int, default=256) : the batch size for training
        - num_timesteps (int, default=14) : the number of sequential inputs per data instance
        
        RETURNS
        -------
        - None
        """
        # read in and normalize data ; Refer "Statistically Sound Machine Learning For Algorithmic Trading"
        try:
            data = pd.read_csv(data_path).sort_values('Date')
            data['Q25'] = data['Close'].rolling(window=200).quantile(0.25)
            data['Q50'] = data['Close'].rolling(window=200).quantile(0.5)
            data['Q75'] = data['Close'].rolling(window=200).quantile(0.75)
            data['Close'] = norm.cdf(0.5*(data['Close']-data['Q50'])/(data['Q75']-data['Q25']))
            data = data['Close'].dropna()
            data = data.values[:, np.newaxis]
        except:
            data = pd.read_csv(data_path).sort_values('date')
            data['Q25'] = data['close'].rolling(window=200).quantile(0.25)
            data['Q50'] = data['close'].rolling(window=200).quantile(0.5)
            data['Q75'] = data['close'].rolling(window=200).quantile(0.75)
            data['close'] = norm.cdf(0.5*(data['Close']-data['Q50'])/(data['Q75']-data['Q25']))
            data = data['close'].dropna()
            data = data.values[:, np.newaxis]
        self.batch_size, self.input_steps, self.output_steps = batch_size, input_steps, output_steps
        N, D = data.shape
        self.split_idx = int(N*train_size)
        self.x = np.zeros((N-(input_steps+output_steps)-3, input_steps, 1))
        self.y = np.zeros((N-(input_steps+output_steps)-3, input_steps, output_steps))
        for idx in range(N-(input_steps+output_steps)-3):
            self.x[idx] = data[idx:idx+input_steps]
            target_idx = idx+1
            temp = data[target_idx:target_idx+input_steps+output_steps] 
            for t_idx in range(input_steps):
                self.y[idx, t_idx, :] = temp[t_idx:t_idx+output_steps].T[0]
        self.y = self.y[:, -1, :] # shape of targets : (N, 1)
  
    def sample_batches(self, num_batches=72):
        """
        Shuffles the model's input (and output) data and returns batches of data for training.
        
        PARAMETERS
        ----------
        - num_batches (int, default=72) : the number of batches to sample
        
        RETURNS
        -------
        - Numpy arrays containing the training batch of shape (num_batches, batch_size, input_steps, output_steps) 
          and corresponding targets of shape (num_batches, batch_size, output_steps).
        """
        assert self.batch_size<=self.split_idx, 'Not enough data points to sample a batch!'
        x_batches = np.zeros(shape=[num_batches, self.batch_size, self.input_steps, self.output_steps])
        y_batches = np.zeros(shape=[num_batches, self.batch_size, self.output_steps])
        for batch_idx in range(num_batches):
            sampled_idxs = np.random.choice(np.arange(self.split_idx), size=self.batch_size)
            x_batches[batch_idx] = self.x[sampled_idxs]
            y_batches[batch_idx] = self.y[sampled_idxs]
        return x_batches, y_batches # default shapes are (72, 256, 28, 1) and (72, 256, 1) respectively
    
    def get_test_data(self):
        """
        Returns test data.
        
        PARAMETERS
        ----------
        - None
        
        RETURNS
        -------
        - Numpy arrays containing the test samples of shape (N, input_steps, output_steps) 
          and targets of shape (N, output_steps).
        """
        return self.x[self.split_idx:], self.y[self.split_idx:] # default shapes are (N, 28, 1) and (N, 1) respectively

if __name__=='__main__':
    path = '../data/PFE.csv'
    helper = DataHelper(path)
    helper.get_test_data()
    x_test, y_test = helper.sample_batches()