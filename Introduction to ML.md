## Introduction to ML
**Training** means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.

**Inference** means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions (y'). 

**Regression vs. classification**  
A regression model predicts continuous values. A classification model predicts discrete values.

## Linear Regression
**Loss Function**  
A L2 Loss for a given example is also called squared error.  
Mean square error (MSE) is the average squared loss per example over the whole dataset.  

[]: # ($$ MSE = \frac{1}{N} \sum_{(x,y)\in D} (y - prediction(x))^2 $$)
<!-- \\(MSE = \frac{1}{N} \sum_{(x,y)\in D} (y-prediction(x))^2 \\) -->
\\[MSE = \frac{1}{N} \sum_{(x,y)\in D} (y-prediction(x))^2 \\]

**Gradient Descent**  
A technique to minimize loss by computing the gradients of loss with respect to the model's parameters, conditioned on training data. Informally, gradient descent iteratively adjusts parameters, gradually finding the best combination of weights and bias to minimize loss.

**Learning Rate**  
Gradient descent algorithms multiply the gradient by a scalar known as the learning rate (also sometimes called step size) to determine the next point. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.



