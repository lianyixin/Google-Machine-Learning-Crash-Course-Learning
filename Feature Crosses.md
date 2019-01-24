## Feature Crosses

**Encoding Nonlinearity**

A feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term cross comes from cross product.) 

**Crossing One-Hot Vectors**

In practice, machine learning models seldom cross continuous features. However, machine learning models do frequently **cross one-hot feature vectors**. Think of feature crosses of one-hot feature vectors as logical conjunctions.

Linear learners scale well to massive data. Using feature crosses on massive data sets is one efficient strategy for learning highly complex models. Neural networks provide another strategy.

**Overcrossing**

If we use a model that is too complicated, such as one with too many crosses, we give it the opportunity to fit to the noise in the training data, often at the cost of making the model perform badly on test data.

## Regularization: Simplicity

we could prevent overfitting by penalizing complex models, a principle called **regularization**.

we'll now minimize loss+complexity, which is called structural risk minimization:

$$\text{minimize(Loss(Data|Model) + complexity(Model))}$$

Machine Learning Crash Course focuses on two common (and somewhat related) ways to think of model complexity:

Model complexity **as a function of the weights of all the features in the model**.  
Model complexity **as a function of the total number of features with nonzero weights**. (A later module covers this approach.)

We can quantify complexity using the **L2 regularization formula**, which defines the regularization term as the sum of the squares of all the feature weights:

$$L_2\text{ regularization term} = ||\boldsymbol w||_2^2 = {w_1^2 + w_2^2 + ... + w_n^2}$$

Model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as **lambda** (also called the **regularization rate**). That is, model developers aim to do the following:

$$\text{minimize(Loss(Data|Model)} + \lambda \text{ complexity(Model))}$$

**Choosing a lambda value**, the goal is to strike the **right balance between simplicity and training-data fit**:

If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.

If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

**Learning Rate and Lambda**

There's a close connection between learning rate and lambda. Strong L2 regularization values tend to drive feature weights closer to 0. Lower learning rates (with early stopping) often produce the same effect because the steps away from 0 aren't as large. Consequently, tweaking learning rate and lambda simultaneously may have confounding effects.

Early stopping means ending training before the model fully reaches convergence. In practice, we often end up with some amount of implicit early stopping when training in an online (continuous) fashion. That is, some new trends just haven't had enough data yet to converge.

As noted, the effects from changes to regularization parameters can be confounded with the effects from changes in learning rate or number of iterations. **One useful practice (when training across a fixed batch of data) is to give yourself a high enough number of iterations that early stopping doesn't play into things**.

## Regularization: Sparsity

In a high-dimensional sparse vector, **it would be nice to encourage weights to drop to exactly 0 where possible**. A weight of exactly 0 essentially removes the corresponding feature from the model. Zeroing out features will save RAM and may reduce noise in the model.

Would L2 regularization accomplish this task? Unfortunately not. L2 regularization encourages weights to be small, but doesn't force them to exactly 0.0.

**L0 regularization** (An alternative idea would be to try and create a regularization term that penalizes the count of non-zero coefficient values in a model.) would turn our convex optimization problem into a non-convex optimization problem that's NP-hard.

However, there is a **regularization term called L1 regularization** that serves as an approximation to L0, but has the advantage of being convex and thus efficient to compute. So we can use L1 regularization to encourage many of the uninformative coefficients in our model to be exactly 0, and thus reap RAM savings at inference time.

**L1 regularization—penalizing the absolute value of all the weights—turns out to be quite efficient for wide models.**

**Why L1**

You can think of the derivative of L2 as a force that removes x% of the weight every time. As Zeno knew, even if you remove x percent of a number billions of times, the diminished number will still never quite reach zero. (Zeno was less familiar with floating-point precision limitations, which could possibly produce exactly zero.) At any rate, L2 does not normally drive weights to zero.

You can think of the derivative of L1 as a force that subtracts some constant from the weight every time. However, thanks to absolute values, L1 has a discontinuity at 0, which causes subtraction results that cross 0 to become zeroed out. For example, if subtraction would have forced a weight from +0.1 to -0.2, L1 will set the weight to exactly 0. Eureka, L1 zeroed out the weight.



