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
The ideal learning rate in one-dimension is $$ \frac{ 1 }{ f(x)''} $$ (the inverse of the second derivative of f(x) at x).  
The ideal learning rate for 2 or more dimensions is the inverse of the **Hessian** (matrix of second partial derivatives).

In practice, finding a "perfect" (or near-perfect) learning rate is not essential for successful model training. The goal is to find a learning rate large enough that gradient descent converges efficiently, but not so large that it never converges.

**Stochastic gradient descent**  
By choosing examples at random from our data set, we could estimate (albeit, noisily) a big average from a much smaller one. Stochastic gradient descent (SGD) takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at random.

**Mini-batch stochastic gradient descent (mini-batch SGD)**  
It is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.


