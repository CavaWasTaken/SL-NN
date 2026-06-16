# Statistical Learning and Neural Networks
### A.A. 2022/2023 — Computer Labs

A collection of Python lab notebooks covering classical machine learning, dimensionality reduction, probabilistic filtering, and deep learning with TensorFlow/Keras.

---

## Labs Overview

### Lab 1 — k-NN Classifier
Implements a k-Nearest Neighbors classifier **from scratch** (NumPy only) and evaluates it across three datasets:

- **Synthetic dataset** — binary classification; accuracy plotted vs. k for both train and test sets, with analysis of overfitting/underfitting.
- **Wine quality dataset** — multi-class classification (10 classes, 11 features); also explored via **linear regression** with MSE evaluation.
- **Phoneme dataset** — 5-class classification over 256 features; includes **4-fold cross-validation**.

Key concepts: Minkowski distance, majority voting, train/test split, cross-validation.

---

### Lab 2 (Classical) — k-NN continued / Linear Regression
Extension of Lab 1 Part 2. Uses `sklearn.linear_model.LinearRegression` for wine quality regression and compares accuracy and MSE against k-NN.

---

### Lab 3 — Principal Component Analysis (PCA)
Applied to a **NASA AVIRIS hyperspectral image** of Indian Pines (145×145 pixels, 220 spectral bands).

- Covariance matrix estimation and eigendecomposition
- MSE reconstruction error vs. number of components K
- **Optional Exercise 2:** Linear Discriminant Analysis (LDA) on raw, PCA-reduced, whitened-PCA, and random-feature-reduced data; comparison with an **SVM** (100% accuracy)

Key concepts: standardization, eigenvector sorting, whitening, kernel trick.

---

### Lab 4 — Kalman Filter
Tracks pedestrian trajectories from a real crowd dataset (`crowds_students001`).

- Simulates noisy observations by adding Gaussian noise to ground-truth positions
- Implements a **constant-velocity Kalman filter** (4D state: x, y, vx, vy)
- Explores the effect of σ_R (measurement noise) and σ_Q (process noise) on tracking quality

Key concepts: predict/update cycle, Kalman gain, covariance propagation.

---

### Lab 1 (Deep Learning) — CNN for Digit Recognition
Trains a **Convolutional Neural Network** on MNIST using TensorFlow/Keras.

- Architecture: 3× Conv2D + BatchNorm + ReLU + MaxPooling, Dropout, Dense (softmax)
- Optimizer: Adam; Loss: sparse categorical cross-entropy
- Test accuracy: ~97.6%

Key concepts: convolutional encoding, batch normalization, dropout, gradient descent, Adam.

---

### Lab 2 (Deep Learning) — Transfer Learning (Cats vs. Dogs)
Fine-tunes **MobileNetV2** (pretrained on ImageNet) for binary image classification.

- Custom data loader with `tf.data.Dataset`
- Frozen base model + `GlobalAveragePooling2D` + Dense(2, softmax) head
- Training accuracy: ~99.9% after 5 epochs

Key concepts: transfer learning, feature extraction, GlobalAveragePooling.

---

### Lab 3 (Deep Learning) — Image Segmentation & Quantization
Semantic segmentation of **Sentinel-2 satellite images** (256×256, 12 spectral bands) to identify cultivated land.

- Implements a **U-Net** with skip connections, encoder/decoder structure, UpSampling2D
- **Post-Training Quantization (PTQ)** to TFLite (float32 → int8), with custom evaluation loop
- Attempted **Quantization-Aware Training (QAT)** (limited by BatchNorm layer compatibility)

Key concepts: U-Net, skip connections, image segmentation, model quantization, TFLite.

---

## Requirements

```
numpy
matplotlib
h5py
scipy
scikit-learn
tensorflow >= 2.x
tensorflow_model_optimization
ndjson
pandas
seaborn
```

---

## Datasets

| Dataset | Source |
|---|---|
| Synthetic / Wine / Phoneme | Provided as `.hdf5` files |
| Indian Pines (hyperspectral) | [AVIRIS / EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Pedestrian trajectories | [TrajNet — crowds_students001](https://paperswithcode.com/dataset/trajnet-1) |
| MNIST | `tf.keras.datasets.mnist` |
| Cats vs. Dogs | Provided as JPEG files |
| Sentinel-2 | Provided as `.npz` file |

---

## Repository Structure

```
.
├── Lab1_kNN/
│   └── Lab1.ipynb
├── Lab2_LinearRegression/
│   └── Lab2.ipynb
├── Lab3_PCA/
│   └── Lab3.ipynb
├── Lab4_KalmanFilter/
│   └── Lab4.ipynb
├── DL_Lab1_CNN/
│   └── lab1.ipynb
├── DL_Lab2_TransferLearning/
│   └── lab2.ipynb
├── DL_Lab3_Segmentation/
│   └── lab3.ipynb
└── README.md
```
