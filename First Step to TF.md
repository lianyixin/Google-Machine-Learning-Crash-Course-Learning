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



**有适用于模型调整的标准启发法吗？**  
这是一个常见的问题。简短的答案是，不同超参数的效果取决于数据。因此，不存在必须遵循的规则，您需要对自己的数据进行测试。  
即便如此，我们仍在下面列出了几条可为您提供指导的经验法则：  

训练误差应该稳步减小，刚开始是急剧减小，最终应随着训练收敛达到平稳状态。  
如果训练尚未收敛，尝试运行更长的时间。  
如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。  
但有时如果学习速率过高，训练误差的减小速度反而会变慢。    
如果训练误差变化很大，尝试降低学习速率。  
**较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。**  
**批量大小过小也会导致不稳定情况。**不妨先尝试 100 或 1000 等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。
重申一下，切勿严格遵循这些经验法则，因为效果取决于数据。请始终进行试验和验证。

**合成特征和离群值**  
学习目标：  
创建一个合成特征，即另外两个特征的比例  
将此新特征用作线性回归模型的输入  
通过**识别和截取（移除）输入数据中的离群值**来提高模型的有效性