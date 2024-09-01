# Apple Stock Price Prediction using LSTM
# Project Overview
This project demonstrates a time series forecasting approach for predicting Apple's stock prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock price data to predict future closing prices.

## Table of Contents
- [Apple Stock Price Prediction using LSTM](#apple-stock-price-prediction-using-lstm)
- [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview-1)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training the Model](#training-the-model)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Results](#results)

## Project Overview
The goal of this project is to predict Apple's stock closing prices using historical data. We leverage an LSTM neural network, which is particularly suited for sequence prediction tasks like time series forecasting.

## Data Preparation
1. **Downloading the Data**:
- We use `yfinance` to download the historical stock prices of Apple (`AAPL`).

2. **Preprocessing**:
- We focus on the Close price, discarding other columns.
- Lagged features are generated to capture the time dependency of stock prices. We use a sequence length of 7 (i.e., the past week's prices) as input features.
- We filter out data before January 1, 2005, as the prices before this date show little variation.
- The data is normalized using `MinMaxScaler` to improve the performance of the neural network.

1. **Data Splitting**:
- The data is split into training and testing sets, with the last 182 days reserved for testing.

1. **Tensor Conversion**:
- The data is converted into `torch.Tensor` format, reshaped appropriately for input to the LSTM model.

## Model Architecture
The LSTM model is implemented in PyTorch with the following structure:
- **Input Layer**: The input sequence has one feature (the closing price).
- **LSTM Layers**: The model consists of 5 LSTM layers with 128 hidden units.
- **Output Layer**: A fully connected linear layer that outputs the predicted stock price.

## Training the Model
- **Loss Function**: Mean Squared Error (MSE) is used to measure the model's performance.
- **Optimizer**: The Adam optimizer is used with a learning rate of 1e-4.
- **Training**: The model is trained for several epochs, adjusting weights to minimize the loss function.

## Dependencies
The project requires the following Python packages:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `yfinance`
- `scikit-learn`
- `torch`
- `torchtyping`

You can install the necessary packages using pip:
```bash
pip install numpy pandas seaborn matplotlib yfinance scikit-learn torch torchtyping
```

## Usage
1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2. **Run the Jupyter notebook**: Open the notebook in Jupyter or Google Colab and run all cells to execute the data preparation, model training, and evaluation steps.

## Results
The trained LSTM model is capable of predicting Apple's stock prices based on the patterns observed in the historical data. The notebook includes visualization of both the training process and the predictions on the test set.