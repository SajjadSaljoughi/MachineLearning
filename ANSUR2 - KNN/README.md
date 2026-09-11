# K-Nearest Neighbors (KNN) — Exercise 2

This project is my **second Machine Learning exercise**, focused on understanding and implementing the **K-Nearest Neighbors (KNN)** classification algorithm from scratch.

The main goal of this project is to understand how KNN works internally and compare my own implementation with the implementation provided by **Scikit-Learn**.

---

## Project Overview

In this project, I used the **ANSUR II dataset** to build a simple classification model that predicts a person's **gender** based on two physical measurements:

* **Weight**
* **Height**

The dataset contains measurements from both male and female participants.

The complete dataset contains **6,068 samples**:

* 4,982 male samples
* 1,986 female samples

The dataset was divided into:

* **80% Training Data:** 4,854 samples
* **20% Test Data:** 1,214 samples

The same train/test split was used to evaluate both implementations.

---

## Dataset

The project uses the **ANSUR II** dataset, which contains a large collection of anthropometric measurements.

For this exercise, only two features were selected:

```text
Features (X):
    - Weight (kg)
    - Height (cm)

Target (Y):
    - Gender
```

Gender labels were converted into numerical values:

```text
Female → 0
Male   → 1
```

The original measurements were also converted into appropriate units before training.

---

## KNN Implementation

Instead of directly using an existing Machine Learning library, I implemented the KNN algorithm from scratch using **NumPy**.

The implementation includes the following main components:

### 1. Euclidean Distance

The distance between two data points is calculated using the Euclidean distance formula:

```text
distance = √Σ(x₁ - x₂)²
```

### 2. Training

KNN is a lazy learning algorithm, so the training process simply stores the training data and labels.

### 3. Prediction

For each test sample:

1. Calculate its distance from every training sample.
2. Sort the distances.
3. Select the `k` nearest neighbors.
4. Find the most common class among those neighbors.
5. Use that class as the prediction.

### 4. Evaluation

The model accuracy is calculated by comparing predicted labels with the actual labels.

The custom implementation contains `fit`, `predict`, and `evaluate` methods.

---

## K Values

To investigate the effect of different values of `k`, I tested:

```text
k = 3
k = 5
k = 7
k = 9
k = 11
```

Each value was evaluated using both:

* My custom KNN implementation
* Scikit-Learn's `KNeighborsClassifier`

---

## Results

The following table shows the accuracy obtained on the test dataset:

|  K | My KNN Accuracy | Scikit-Learn KNN Accuracy |
| -: | --------------: | ------------------------: |
|  3 |          80.56% |                    80.72% |
|  5 |          81.96% |                    81.88% |
|  7 |          82.13% |                    82.04% |
|  9 |          83.03% |                    83.03% |
| 11 |      **83.44%** |                **83.44%** |

The results above are taken from the actual experiment in the notebook.

---

## Comparison

The results show that the custom implementation performs very similarly to the Scikit-Learn implementation.

An interesting observation is that the accuracy generally improves as `k` increases from **3 to 11**.

The best accuracy in this experiment was achieved with:

```text
k = 11
Accuracy = 83.44%
```

Both implementations achieved the same accuracy for `k = 9` and `k = 11`.

This close similarity suggests that the core logic of my KNN implementation is working correctly.

---

## Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-Learn**
* **Jupyter Notebook**

---

## Project Structure

```text
ANSUR2 - KNN/
│
├── data/
│   ├── ANSUR II FEMALE Public.csv
│   └── ANSUR II MALE Public.csv
│
├── knn.py
├── main.ipynb
└── README.md
```

---

## What I Learned

Through this exercise, I practiced:

* Understanding the KNN algorithm
* Implementing KNN from scratch
* Calculating Euclidean distance
* Working with NumPy arrays
* Preparing and preprocessing a dataset
* Splitting data into training and testing sets
* Evaluating classification accuracy
* Testing different values of `k`
* Comparing a custom Machine Learning implementation with Scikit-Learn
* Working with confusion matrices and classification results

---

## Conclusion

This project helped me understand KNN beyond simply using a Machine Learning library.

By implementing the algorithm from scratch and comparing its results with Scikit-Learn, I was able to verify the behavior of my implementation and gain a better understanding of how KNN makes predictions.

The final result was an accuracy of approximately **83.44%** with `k = 11`.

---

## Author

**Sajjad Saljoughi**

This project is part of my journey to learn **Machine Learning and Artificial Intelligence with Python**.
