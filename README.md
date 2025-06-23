# 🧠 MNIST Neural Network from Scratch

> A simple yet effective neural network built entirely from scratch in **pure Python** — no deep learning libraries like TensorFlow or PyTorch used. Trained on the MNIST dataset, the model achieves ~90% accuracy using a single hidden layer, softmax output, and backpropagation with optional L2 regularization.

---

## 🚀 Features

- Pure Python implementation (no ML libraries)
- Manual forward and backward propagation
- Configurable hidden layer size
- L2 regularization support
- Softmax output layer
- Gradient descent optimizer
- ~90% accuracy on MNIST test set

---

## 🧾 Dataset

The project uses the [MNIST in CSV format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) dataset. (The dataset is also available as a zip file which can be downloaded from the files section.)

---

## ⚙️ How It Works

### 🛠 Training Flow

#### 1. Data Preprocessing:
  - Load and shuffle MNIST data
  - Normalize pixel values (0–255 → 0–1)
  - Split into training, dev, and test sets

#### 2. Network Architecture:
  - Input Layer: 784 neurons (28x28 image)
  - Hidden Layer: Configurable (e.g. 110 neurons)
  - Output Layer: 10 neurons (digits 0–9)

#### 3. Activations:
  - Hidden Layer: ReLU
  - Output Layer: Softmax

#### 4. Training
  - Forward pass → Loss → Backpropagation
  - Parameter updates via gradient descent
  - Optional L2 regularization

---

## 📊 Model Evaluation

Use the following function to evaluate performance on any dataset:

```
evaluate_model(W1, b1, W2, b2, X_test, Y_test)
```

---

## 🧪 Hyperparameter Tuning

### 🔹 Tune Hidden Layer Size

```
find_best_hidden_size(
    X_train, Y_train,
    X_dev, Y_dev,
    alpha=0.1,
    iterations=3000,
    lambd=0,
    hidden_sizes=[64, 110, 128]
)
```

### 🔹 Tune Regularization Strength (λ)

```
find_best_lambda(
    X_train, Y_train,
    X_test, Y_test,
    alpha=0.1,
    iterations=501,
    lambdas=[0.01, 0.1, 1],
    hidden_size=110
)
```

---

## 🖼️ Sample Predictions

Use this function to view a prediction and image side-by-side:

```
test_prediction(index, W1, b1, W2, b2)
```
Example:

```
test_prediction(0, W1, b1, W2, b2)
```

---

## 📈 Accuracy

| Dataset  | Accuracy |
| -------- | -------- |
| Dev Set  | \~89–91% |
| Test Set | \~90%    |

---

## 📚 Educational Value

This project is ideal if you want to:

  - Understand how neural networks work under the hood
  - Learn backpropagation and gradient descent without abstraction
  - Explore basics of tuning and regularization

---
