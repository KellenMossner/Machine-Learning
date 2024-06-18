# Naive Bayes Classifier

## Description
This repository contains an implementation of a Naive Bayes Classifier using first principles, taught in Computer Science 315, implemented with numpy. The Naive Bayes Classifier is a popular machine learning algorithm used for classification tasks. The classifier is based on the Naive Bayes theorem, which assumes that the features are conditionally independent given the class label.

## Features
- Easy to understand and implement
- Efficient for large datasets
- Performs well in many real-world scenarios

## Usage
1. Import the NaiveBayesClassifier class from the `naive_bayes.py` module.
2. Create an instance of the classifier: `classifier = NaiveBayes()`
3. Train the classifier using labeled training data: `classifier.fit(X_train, y_train)`
4. Predict the class labels for new instances: `predictions = classifier.predict(X_test)`
5. Evaluate the accuracy of the classifier: `accuracy = classifier.evaluate(X_test, y_test)`