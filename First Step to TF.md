## First Step to Tensorflow
The following figure shows the current hierarchy of TensorFlow toolkits: 
 
![Hierarchy of TF](https://developers.google.cn/machine-learning/crash-course/images/TFHierarchy.svg)

**TensorFlow** consists of the following two components:  
a graph protocol buffer  
a runtime that executes the (distributed) graph

TensorFlow 的名称源自**张量**，张量是任意维度的数组。借助 TensorFlow，您可以操控具有大量维度的张量。即便如此，在大多数情况下，您会使用以下一个或多个低维张量：

标量是零维数组（零阶张量）。例如，\'Howdy\' 或 5。  
矢量是一维数组（一阶张量）。例如，[2, 3, 5, 7, 11] 或 [5]  
矩阵是二维数组（二阶张量）。例如，[[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]   
 
TensorFlow **指令**会创建、销毁和操控张量。典型 TensorFlow 程序中的大多数代码行都是指令。  

TensorFlow **图**（也称为计算图或数据流图）是一种图数据结构。很多 TensorFlow 程序由单个图构成，但是 TensorFlow 程序可以选择创建多个图。图的节点是指令；图的边是张量。张量流经图，在每个节点由一个指令操控。一个指令的输出张量通常会变成后续指令的输入张量。TensorFlow 会实现延迟执行模型，意味着系统仅会根据相关节点的需求在需要时计算节点。

**张量**可以作为常量或变量存储在图中。您可能已经猜到，常量存储的是值不会发生更改的张量，而变量存储的是值会发生更改的张量。不过，您可能没有猜到的是，常量和变量都只是图中的一种指令。常量是始终会返回同一张量值的指令。变量是会返回分配给它的任何张量的指令。


```python
import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier(feature_columns)

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)
```

Common hyperparameters in Machine Learning Crash Course exercises  
Many of the coding exercises contain the following hyperparameters:

**steps**, which is the total number of training iterations. One step calculates the loss from one batch and uses that value to modify the model's weights once.  
**batch size**, which is the number of examples (chosen at random) for a single step. For example, the batch size for SGD is 1.  
The following formula applies:  
$$total\,number\,of\,trained\,examples = batch\,size * steps$$

**periods**, which controls the granularity of reporting. For example, if periods is set to 7 and steps is set to 70, then the exercise will output the loss value every 10 steps (or 7 times).   
$$number\,of\,training\,examples\,in\,each\,period = \frac{batch\,size * steps} {periods}$$

