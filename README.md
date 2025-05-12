# Multiclass Logistic Regression for Digit Recognition (NumPy Implementation)

This project implements a **multiclass logistic regression model from scratch** using only NumPy. It is used to recognize handwritten digits (0–9) from the MNIST dataset.

## Key Features

- Implements binary and multiclass logistic regression without using any machine learning libraries.
- Reads raw MNIST `.idx` files directly.
- Supports training, prediction, accuracy evaluation, and model export (`.npz`).
- Entirely based on NumPy and built from first principles.

## Project Structure
.  
├── train.py    # Main script: load data, train model, export model  
├── scripts.py  # Core logic: logistic regression, training, prediction  
├── data/       # MNIST dataset files (.idx format)  
├── model.npz   # Saved model weights (after training)  
├── output.py   # Testing model (using weight in model.npz)  
└── README.md


## How It Works

### 1. Data Loading
The `read_idx()` function loads MNIST data from raw `.idx` files into NumPy arrays.

### 2. Model
A one-vs-all strategy is used for multiclass classification:
- Each digit class (0–9) is trained with a separate binary logistic regression model.
- During prediction, each model produces a probability via sigmoid.
- Final output uses a softmax-like method to choose the most probable class.

### 3. Training
The training logic is built inside the `Z` class (for binary) and `MultinomialLogistic` (for multiclass):

```python
# file train.py
A = MultinomialLogistic(X, Y)
A.train(learning_rate=0.05, epochs=10000)  #<---- Can modify
A.to_file("model.npz")
```

### Tesing
The file `output.py` allows the user to draw and write a digit on the window, then press OK. The result will be returned and printed out.

<p align="center">
  <img src="demo.png" alt="Digit Drawing Demo" width="300"/>
</p>