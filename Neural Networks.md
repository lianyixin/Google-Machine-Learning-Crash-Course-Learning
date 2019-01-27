## Neural Networks

**Activation Functions**

To model a nonlinear problem, we can directly introduce a nonlinearity. We can pipe each hidden layer node through a **nonlinear function**.

In the model represented by the following graph, the value of each node in Hidden Layer 1 is transformed by a nonlinear function before being passed on to the weighted sums of the next layer. This nonlinear function is called the activation function.

![network](https://developers.google.cn/machine-learning/crash-course/images/activation.svg)

**Common Activation Functions**

The following sigmoid activation function converts the weighted sum to a value between 0 and 1.

$$F(x)=\frac{1} {1+e^{-x}}$$

The following rectified linear unit activation function (or ReLU, for short) often works a little better than a smooth function like the sigmoid, while also being significantly easier to compute.

$$F(x)=max(0,x)$$

**The superiority of ReLU is based on empirical findings, probably driven by ReLU having a more useful range of responsiveness. A sigmoid's responsiveness falls off relatively quickly on both sides.**

In fact, any mathematical function can serve as an activation function. Suppose that sigma represents our activation function (Relu, Sigmoid, or whatever). Consequently, the value of a node in the network is given by the following formula:

$$\sigma(\boldsymbol w \cdot \boldsymbol x+b)$$

## Training NN

This section explains backpropagation's failure cases and the most common way to regularize a neural network.

###Failure Cases

**Vanishing Gradients**

The gradients for the lower layers (closer to the input) can become very small. In deep networks, computing these gradients can involve taking the product of many small terms.

When the gradients vanish toward 0 for the lower layers, these layers train very slowly, or not at all.

The **ReLU activation function** can help prevent vanishing gradients.

**Exploding Gradients**

If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms. In this case you can have exploding gradients: gradients that get too large to converge.

**Batch normalization** can help prevent exploding gradients, as can lowering the learning rate. (Batch normalisation keeps the input to each intermediate layer normalised, which removes the dependence of the gradients on the scaling of the intermediate-outputs within the network.)

**Dead ReLU Units**

Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck. It outputs 0 activation, contributing nothing to the network's output, and gradients can no longer flow through it during backpropagation. With a source of gradients cut off, the input to the ReLU may not ever change enough to bring the weighted sum back above 0.

**Lowering the learning rate** can help keep ReLU units from dying.

**Dropout Regularization**

Yet another form of regularization, called Dropout, is useful for neural networks. It works by randomly "dropping out" unit activations in a network for a single gradient step. The more you drop out, the stronger the regularization:

0.0 = No dropout regularization.  
1.0 = Drop out everything. The model learns nothing.  
Values between 0.0 and 1.0 = More useful.




