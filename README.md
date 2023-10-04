##stockPredict
Implementation of stock price prediction using LSTM with PyTorch

##Software Environment
Python 3.0 or later
PyTorch 1.3.1
torchvision 0.4.1
Pillow 7.1.2
pandas 1.0.3
Project Structure
Project Structure

##data directory: CSV files of the Shanghai Stock Exchange index
model directory: Saved model files
dataset.py: Data loading and preprocessing class, including data normalization, splitting into training and testing sets, etc.
evaluate.py: Prediction
LSTMModel.py: Definition of the LSTM model
parsermy.py: Common parameters
train.py: Model training

##How to Run:
Run train.py directly to start model training.
Run evaluate.py directly to start model prediction.
