# StockPricePredictor

An LSTM-based Deep Neural Network to predict a sequence of stock prices from an input sequence. Processes the input using some 'k' fully-connected layers to project it into a space that allows for a better interpretation of temporal dynamics. The processed input is fed into some 'l' LSTM-layers and then the into some 'm' final output layers.

Input shape : (N, T, 1), Output shape : (N, T, T') where T' is the number of sequential timesteps per output sequence.

Python Dependencies:
-------------------
- tensorflow
- pandas

