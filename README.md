# StockPricePredictor

An LSTM-based Deep Neural Network to predict a sequence of stock prices from an input sequence. Processes the input using some 'k' fully-connected layers to project it into a space that allows for a better interpretation of temporal dynamics. The processed input is fed into some 'l' LSTM-layers and then into some 'm' final output layers.

Input shape : (N, T, 1), Output shape : (N, 1) which are the outputs at the (T+1)st timestep for every input sequence.

Python Dependencies:
-------------------
- tensorflow
- pandas

