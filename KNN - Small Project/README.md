# KNN Visualization Project (From Scratch)

## Overview

This project is my first small machine learning project, focused on understanding and implementing the **K-Nearest Neighbors (KNN)** algorithm from scratch using Python.

The main goal of this project is to deeply understand how KNN works internally, without relying on machine learning libraries such as scikit-learn.

In this project, I created a simple dataset representing clothing items and used KNN to classify them based on their physical dimensions.

---

## Project Idea

We simulate three types of clothing items:

* Shirts
* Pants
* Hats

Each clothing item is represented using two features:

* Width
* Height

Using these features, the KNN algorithm predicts the class of a new clothing item.

This project also includes visualization to clearly demonstrate how different classes are separated in feature space.

---

## What This Project Demonstrates

This project demonstrates:

* Implementing KNN from scratch
* Understanding Euclidean distance
* Generating synthetic datasets using NumPy
* Data visualization using Matplotlib
* Basic classification logic
* How machine learning makes decisions based on distance

---

## How the Algorithm Works

The KNN algorithm follows these steps:

1. Store all training data points with their labels.
2. Choose a new data point to classify.
3. Calculate the distance between the new point and all training points.
4. Select the K nearest neighbors.
5. Count the most frequent label among the neighbors.
6. Assign that label as the prediction.

Distance is calculated using the Euclidean distance formula:

[
distance = \sqrt{\sum (x_1 - x_2)^2}
]

---

## Visualization

The dataset is visualized using a scatter plot:

* Red → Shirts
* Blue → Pants
* Green → Hats

This helps visually understand how KNN separates classes based on proximity.

---

## Example Prediction

Example new clothing item:

```
Width: 60
Height: 110
```

The algorithm calculates distances and predicts the most likely class based on nearest neighbors.

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Project Structure

```
project/
│
├── main.ipynb
├── images/
│   ├── shirt.jpg
│   ├── pants.jpg
│   └── hat.jpg
│
└── README.md
```

---

## Learning Objectives

This project helped me learn:

* Core machine learning concepts
* How KNN works internally
* Distance-based classification
* Data visualization
* Writing clean and understandable code
* Building ML algorithms without external libraries

---

## Why This Project Matters

Instead of using ready-made libraries, implementing KNN from scratch provides a deeper understanding of:

* How machine learning algorithms actually work
* The mathematics behind classification
* How data influences predictions

This is an important foundational step before using advanced ML frameworks.

---

## Future Improvements

Possible future enhancements:

* Add interactive visualization
* Allow dynamic selection of K value
* Compare results with scikit-learn implementation
* Add more features (weight, color, etc.)
* Build a GUI version

---

## Author

This project was created as part of my machine learning learning journey, starting from fundamentals and building algorithms step by step.

---

## Final Note

This is a beginner-level machine learning project, but it builds strong intuition about how classification algorithms work internally. Understanding these fundamentals is essential before moving to more advanced ML and AI topics.
