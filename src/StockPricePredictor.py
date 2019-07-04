#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:39:37 2019

@author: roshanprakash
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from DataHelper import *

class PricePredictor:
   
    """ LSTM Network to predict stock prices. """
    
    def __init__(self, num_timesteps=28, num_processing_layers=2, process_units=[256, 84], num_LSTM_layers=5, out_dims=[512, 256, 128, 64, 16],\
                 num_output_layers=3, output_units=[64, 12, 1], drop_rate=0.2, learning_rate=0.001):
        """
        Initializes the network architecture
        
        PARAMETERS
        ----------
        - num_timesteps (int, default=28) : the number of timesteps (sequential inputs for the LSTM model)
        - num_processing_layers (int, default=2) : the number of fully-connected, initial processing layers 
        - process_units (list, default=[256, 84]) : the number of hidden units in the input processing layers 
        - num_LSTM_layers (int, default=5) : the number of LSTM layers 
        - out_dims (list, default=[512, 256, 128, 64, 16]) : the output dimensions of the LSTM layers
        - num_output_layers (int, default=3) : the number of output layers that process the LSTM output
        - output_units (list, default=[64, 12, 7]) : the number of hidden units in the output processing layers 
        - drop_rate (float, default=0.5) : the dropout rate for the dropout layers
        - learning_rate (float, default=0.001) : the learning rate for network
        
        RETURNS
        -------
        - None.
        """
        # network architecture : input(N,T,1)-->{FC_1, FC_2, ...}-->{(LSTM_1, Batch_Normalize_1, Activate_1, Dropout_1), (.), ..}-->{FC_1, FC_2, ...}-->output(N,T,1)
        assert process_units[-1]%num_timesteps==0, 'Output from processing layers cannot be reshaped before feeding to LSTM layer(s)!'
        assert (num_processing_layers==len(process_units)) and (num_LSTM_layers==len(out_dims)), 'Output dimension must be specified for every layer!'
        self.t = num_timesteps
        # initialize placeholders 
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, num_timesteps, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, output_units[-1]])
        self.is_training = tf.placeholder(tf.bool)
        N = tf.shape(self.x)[0]
        # Network architecture build
        self.layers = {}
         # first few layers are fully-connected layers to preprocess/project the input into a space with easier temporal dynamics
        for layer_idx in range(1,  num_processing_layers+1):
            self.layers['layer_{}'.format(layer_idx)] = tf.keras.layers.Dense(process_units[layer_idx-1], activation=tf.nn.relu)
        # LSTM layers
        for layer_idx in range(num_processing_layers+1, num_processing_layers+num_LSTM_layers+1):
            out_dim = out_dims[layer_idx-num_processing_layers-1]
            self.layers['layer_{}'.format(layer_idx)] = tf.keras.layers.LSTM(out_dim, return_sequences=True)
            # add batch normalization layers
            self.layers['batch_norm_layer_{}'.format(layer_idx)] = tf.keras.layers.BatchNormalization(fused=True)
            self.layers['activation_layer_{}'.format(layer_idx)] = tf.keras.layers.ReLU()
            self.layers['drop_layer_{}'.format(layer_idx)] = tf.keras.layers.Dropout(rate=drop_rate, noise_shape=(N, self.t, out_dim), seed=int(10000*layer_idx))
        # output layers
        for layer_idx in range(num_processing_layers+num_LSTM_layers+1, num_processing_layers+num_LSTM_layers+num_output_layers+1):
            if layer_idx==num_processing_layers+num_LSTM_layers+num_output_layers:
                activation = None
            else:
                activation = tf.nn.relu
            self.layers['layer_{}'.format(layer_idx)] = tf.keras.layers.Dense(output_units[layer_idx-(num_processing_layers+num_LSTM_layers+1)], activation=activation)
        # forward pass
        self.x_ = tf.reshape(self.x, shape=[-1, 1]) # each timestep's value is now considered a single input
        prev_out = None
        for layer_idx in range(1, num_processing_layers+num_LSTM_layers+num_output_layers+1):
            if layer_idx==1:
                prev_out = self.layers['layer_{}'.format(layer_idx)](self.x_)
            else:
                if layer_idx==num_processing_layers+1:
                    # reshape for LSTM layer(s) ; (N*T,D) --> (N,T,D)
                    D = prev_out.shape[-1]
                    prev_out = tf.reshape(prev_out, shape=[-1, self.t, D])
                elif layer_idx==num_processing_layers+num_LSTM_layers+1:
                    # reshape for output layer(s) ; (N,T,D) --> (N*T,D)
                    D = prev_out.shape[-1]
                    prev_out = tf.reshape(prev_out, shape=[-1, D])
                prev_out = self.layers['layer_{}'.format(layer_idx)](prev_out)
                if layer_idx>num_processing_layers and layer_idx<=num_processing_layers+num_LSTM_layers:
                    prev_out = self.layers['batch_norm_layer_{}'.format(layer_idx)](prev_out)
                    prev_out = self.layers['activation_layer_{}'.format(layer_idx)](prev_out)
                    prev_out = self.layers['drop_layer_{}'.format(layer_idx)](prev_out, training=self.is_training)
        # finally, reshape the final output layer's predictions
        self.predictions = tf.reshape(prev_out, shape=[-1, self.t, output_units[-1]])
        # loss computation and weights update
        self.loss = self._compute_normalized_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients, weights = zip(*self.optimizer.compute_gradients(self.loss))
        self.gradients, _ = tf.clip_by_global_norm(gradients, tf.global_norm(gradients)) # normalize gradients and clip
        self.train_step = self.optimizer.apply_gradients(zip(self.gradients, weights))
        
    def _compute_normalized_loss(self):
        """ Normalizes the mean squared error and returns the normalized MSE. """ 
        #loss = tf.losses.mean_pairwise_squared_error(self.y[:, -1, :], self.predictions[:, -1, :])
        loss = tf.losses.mean_squared_error(self.y, self.predictions[:, -1, :])
        return loss
     
    def run(self, datahelper, num_training_epochs=200, print_every=1, save_model=False, save_path=None):
        """ 
        Trains the network and saves the trained model to disk, if required.
        
        PARAMETERS
        ----------
        - datahelper (DataHelper object) : an object of data helper class
        - num_training_epochs (int, default=200) : the number of training epochs
        - print_every (int, default=1) : prints the loss every `print_every` epochs
        - save_model (bool, default=False) : if True, saves trained model to disk
        - save_path (str) : the disk path to save the model
        
        RETURNS
        -------
        - None.
        """
        if save_model and save_path is None:
            raise ValueError('Enter the disk path to save the trained model!')
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            for epoch in range(1, num_training_epochs+1):
                xb, yb = datahelper.sample_batches()
                for idx in range(xb.shape[0]):
                    feed_dict = {self.x:xb[idx], self.y:yb[idx], self.is_training:True}
                    loss, _ = sess.run((self.loss, self.train_step), feed_dict=feed_dict)
                if epoch%print_every==0:
                    print('Completed training {0}/{1} epochs ;'.format(epoch, num_training_epochs))
                    print('--- Loss = {}'.format(loss))
                    print()
                    print('==================================================================================')
                    print()
            print('Model trained successfully! Validating...')
            x_test, y_test = datahelper.get_test_data()
            test_predictions, test_loss = sess.run((self.predictions, self.loss), feed_dict={self.x:x_test, self.y:y_test, self.is_training:False})
            print('Validation-time loss = {}'.format(test_loss))
            if save_model:
                saver.save(sess, save_path)
                print('Saved trained model to disk!')
        # optional plot
        print('Plotting predictions...')
        plt.figure(figsize=(12, 7))
        plt.plot(test_predictions[:, -1, :], color='green', alpha=0.9, label='Predicted Value')
        plt.plot(y_test, color='red', alpha=0.5, label='Ground truth')
        ax = plt.gca()
        ax.grid(color='black', alpha=0.12)
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.title('Predictions vs. Ground truth')
        plt.legend()
        plt.show()
        print('Process complete!')
 
if __name__=='__main__':
    helper = DataHelper('../data/PFE.csv')
    tf.reset_default_graph()
    predictor = PricePredictor(learning_rate=0.0001)
    predictor.run(helper, 100, save_model=True, save_path='../model/model.ckpt')          