# Apple Stock Price Prediction using LSTM
This project demonstrates a time series forecasting approach for predicting Apple's stock prices and traded volume using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock price and volume data to predict future closing prices

## Table of Contents
- [Apple Stock Price Prediction using LSTM](#apple-stock-price-prediction-using-lstm)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training the Model](#training-the-model)
  - [Dependencies](#dependencies)
  - [Usage](#usage)

## Project Overview
The goal of this project is to predict Apple's stock closing prices using historical data. We leverage an LSTM neural network, which is particularly suited for sequence prediction tasks like time series forecasting.

## Data Preparation
1. **Downloading the Data**:
   - Historical stock prices and volume for Apple (`AAPL`) are downloaded using `yfinance`.

2. **Preprocessing**:
   - We focus on the `Close` and `Volume` columns, discarding other data.
   - Lagged features are generated for both `Close` and `Volume`, capturing time dependencies of both stock prices and trading volumes. A sequence length of 7 days (past week's data) is used as input features.
   - Data before January 1, 2005 is excluded due to low variability in prices.
   - The data is normalized using `StandardScaler` to improve the performance of the neural network.

3. **Data Splitting**:
   - The data is split into training and testing sets, with the last 182 days reserved for testing.

4. **Tensor Conversion**:
   - The data is converted into `torch.Tensor` format, reshaped appropriately for input to the LSTM model.

## Model Architecture
The LSTM model is implemented in PyTorch with the following structure:
- **Input Layer**: The input sequence consists of two features: `Close` and `Volume`.
- **LSTM Layers**: The model consists of 5 LSTM layers with 128 hidden units.
- **Output Layer**: A fully connected linear layer that outputs the predicted stock price.

## Training the Model
- **Loss Function**: Mean Squared Error (MSE) is used to measure the model's performance.
- **Optimizer**: The Adam optimizer is used with a learning rate of `3e-5`.
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
    git clone git@github.com:b14ucky/stock-prediction.git
    cd stock-prediction
    ```
2. **Run the Jupyter notebook**: Open the notebook in Jupyter or Google Colab and run all cells to execute the data preparation, model training, and evaluation steps.