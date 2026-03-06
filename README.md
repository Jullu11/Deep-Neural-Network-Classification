# Deep Neural Network Classification

**Author:** Jainesh Lad  
**Dataset:** Bank Marketing (OpenML)

Binary classification project predicting whether a bank client will subscribe to a term deposit based on marketing campaign data, using custom Multi-Layer Perceptron (MLP) classifiers built with TensorFlow's Model Subclassing API.

---

## Overview

This project demonstrates the implementation and evaluation of three custom MLP architectures:

| Model | Description |
|-------|-------------|
| **Simple MLP** | Baseline shallow network |
| **Deep MLP** | Higher capacity deep network |
| **Regularized MLP** | Deep network with dropout and L2 regularization |

---

## Dataset

- **Source:** UCI Machine Learning Repository (via OpenML)
- **Instances:** 45,211 records
- **Features:** 16 input variables (7 numerical, 9 categorical)
- **Target:** Binary — whether the client subscribed to a term deposit (yes/no)
- **Class Distribution:** Imbalanced (~88% no, ~12% yes)

Features include demographic data (age, job, marital status, education), campaign details (contact type, day/month, number of contacts), and economic indicators (employment rate, consumer index, Euribor rate).

---

## Approach

1. Load and explore the Bank Marketing dataset from OpenML
2. Visualize and analyze data distribution, class balance, and feature relationships
3. Preprocess data (encode categorical variables, feature scaling)
4. Implement three custom MLP architectures using TensorFlow Model Subclassing
5. Train and evaluate models with train/validation/test splits
6. Analyze results with comprehensive metrics and visualizations

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

---

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## Usage

1. Open `DNN_Classification.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially (the notebook fetches the dataset from OpenML automatically)

---

## Project Structure

```
.
├── DNN_Classification.ipynb   # Main notebook
├── README.md                   # This file
└── requirements.txt            # Optional dependencies file
```

---

## License

This project is for educational purposes.
