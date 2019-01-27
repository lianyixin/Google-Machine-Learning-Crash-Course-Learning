## Multi-Class Neural Networks

### One vs. All

One vs. all provides a way to leverage binary classification. Given a classification problem with N possible solutions, a one-vs.-all solution consists of N separate binary classifiers—one binary classifier for each possible outcome. During training, the model runs through a sequence of binary classifiers, training each to answer a separate classification question. 

![](https://developers.google.cn/machine-learning/crash-course/images/OneVsAll.svg)

This approach is fairly reasonable when the total number of classes is small, but **becomes increasingly inefficient** as the number of classes rises.

### Softmax

Softmax assigns decimal probabilities to each class in a multi-class problem. Those decimal probabilities must **add up to 1.0**. This additional constraint helps training converge more quickly than it otherwise would.

![softmax](https://developers.google.cn/machine-learning/crash-course/images/SoftmaxLayer.svg)

$$p(y = j|\textbf{x})  = \frac{e^{(\textbf{w}_j^{T}\textbf{x} + b_j)}}{\sum_{k\in K} {e^{(\textbf{w}_k^{T}\textbf{x} + b_k)}} }$$

Note that this formula basically extends the formula for logistic regression into multiple classes.

**Softmax Options**

Consider the following variants of Softmax:

**Full Softmax** is the Softmax we've been discussing; that is, Softmax calculates a probability for every possible class.

**Candidate sampling** means that Softmax calculates a probability for all the positive labels but only for a random sample of negative labels. For example, if we are interested in determining whether an input image is a beagle or a bloodhound, we don't have to provide probabilities for every non-doggy example.

Full Softmax is fairly cheap when the number of classes is small but becomes prohibitively expensive when the number of classes climbs. **Candidate sampling can improve efficiency in problems having a large number of classes**.

### One Label vs. Many Labels

Softmax assumes that each example is a member of exactly one class. Some examples, however, can simultaneously be a member of multiple classes. For such examples:

You may not use Softmax.  
You must rely on multiple logistic regressions.

For example, suppose your examples are images containing exactly one item—a piece of fruit. Softmax can determine the likelihood of that one item being a pear, an orange, an apple, and so on. If your examples are images containing all sorts of things—bowls of different kinds of fruit—then you'll have to use multiple logistic regressions instead.

