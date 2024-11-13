# Cat vs Non-Cat Image Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify images of cats (class 3 in CIFAR-10 dataset) vs non-cats (all other classes). The model is trained on the CIFAR-10 dataset and utilizes binary classification to distinguish between cat and non-cat images.


[Overview](#overview)

[Techniques Covered](#techniques-covered)

[Features](#features)

[Usage](#usage)

[Dependencies](#dependencies)

[Results](#results)


## OverView
The goal of this project is to build a Convolutional Neural Network (CNN) for the classification of cat vs non-cat images using the CIFAR-10 dataset. The model is trained to distinguish between images of cats (class 3 in CIFAR-10) and non-cats (all other classes). The CNN consists of layers such as convolution, pooling, dropout, and fully connected layers, and is evaluated using accuracy, classification reports, and confusion matrices.

## Techniques Covered
- Convolutional Neural Networks (CNN) for binary image classification (cat vs non-cat).
- Data Preprocessing: Normalizing image pixels, reshaping labels, and binary encoding.
- Dropout for regularization to prevent overfitting.
- Model Evaluation: Using accuracy, classification reports, and confusion matrices.
- Visualization: Displaying sample predictions with predicted vs actual labels.

## Features
- Data Preprocessing: Normalizes the image pixels and reshapes them into a 4D array for CNN input. Labels are binary (1 for cat, 0 for non-cat).
- CNN Architecture:
  1. Convolutional layers for feature extraction.
  2. Max pooling for downsampling.
  3. Fully connected layers for classification.
  4. Dropout layers for regularization.
- Model Evaluation: The model’s performance is evaluated using test accuracy, classification report, and confusion matrix.
- Visualizing Predictions: Displays sample predictions with images, predicted, and actual labels.

## Usage
- Load and Preprocess the Data: The CIFAR-10 dataset is loaded, and images are normalized and reshaped. The labels are converted to binary values (1 for cat and 0 for non-cat).
- Build the CNN Model: The CNN model is constructed with layers including convolution, pooling, dropout, and fully connected layers for image classification.
- Train the CNN Model: The model is trained on the preprocessed CIFAR-10 dataset using the training data and evaluated on the test data.
- Evaluate the Model: After training, the model’s performance is evaluated using accuracy, classification reports, and confusion matrices.
- Visualize Predictions: Some predictions from the test set are displayed to check how well the model performs.

## Dependencies
```
tensorflow  # Required for Keras and model training
numpy       # Required for numerical operations
matplotlib  # Required for plotting and visualizations
scikit-learn # Required for classification report and confusion matrix

```
## Results
- Test accuracy: The accuracy of the model after training, evaluated on the test dataset.
- Classification Report & Confusion Matrix: Provides insights into precision, recall, F1 score, and model performance on the test data.
- Visualizations: Displays the model's predictions for some images in the test set.

### Sample Output

#### Test accuracy
```
Test accuracy: 91.48%
```
#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.98      0.95      9000
           1       0.66      0.30      0.41      1000

    accuracy                           0.91     10000
   macro avg       0.79      0.64      0.68     10000
weighted avg       0.90      0.91      0.90     10000
```

#### Confusion Matrix
```
Confusion Matrix:
[[8846  154]
 [ 698  302]]
```

#### Visualized Predictions
[Prediction result for first 5](https://github.com/ZAT90/cat-binary-classification/blob/master/prediction_example.png)
