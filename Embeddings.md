## Embeddings

**Embedding Space**

In general, when learning a d-dimensional embedding, each movie is represented by d real-valued numbers, each one giving the coordinate in one dimension.

Sometimes, we can look at the embeddings and assign semantic meanings to the dimensions, and other times we cannot. Often, each such dimension is called **a latent dimension**, as it represents a feature that is not explicit in the data but rather inferred from it.

**Categorical input data**

Categorical data refers to input features that represent one or more discrete items from a finite set of choices.

Categorical data is most efficiently represented via sparse tensors, which are tensors with very few non-zero elements. 

But however you determine the non-zero values, one-node-per-word gives you very sparse input vectors—very large vectors with relatively few non-zero values. Sparse representations have a couple of problems that can make it hard for a model to learn effectively.

###Drawback

1. Size of network
2. Lack of Meaningful Relations Between Vectors

###Solution: Embedding

The solution to these problems is to use embeddings, **which translate large sparse vectors into a lower-dimensional space that preserves semantic relationships**.

As you can see from the paper exercises, even a small multi-dimensional space provides the freedom to group semantically similar items together and keep dissimilar items far apart. **Position (distance and direction)** in the vector space can encode semantics in a good embedding. 

-> Shrinking the network

-> **Embeddings as lookup tables**

An embedding is a matrix in which each column is the vector that corresponds to an item in your vocabulary. To get the dense vector for a single vocabulary item, you retrieve the column corresponding to that item.

As for a sparse bag of words vector? -> Embedding lookup as matrix multiplication.

###Obtaining Embeddings

**Standard Dimensionality Reduction Techniques**

There are many existing mathematical techniques for capturing the important structure of a high-dimensional space in a low dimensional space. In theory, any of these techniques could be used to create an embedding for a machine learning system. -> For example, **principal component analysis (PCA)** has been used to create word embeddings. Given a set of instances like bag of words vectors, PCA tries to **find highly correlated dimensions that can be collapsed into a single dimension**.

**Word2vec**

Word2vec is an algorithm invented at Google for training word embeddings. Word2vec relies on the **distributional hypothesis** to map semantically similar words to geometrically close embedding vectors.

The distributional hypothesis states that words which often have the same neighboring words tend to be semantically similar. Both "dog" and "cat" frequently appear close to the word "vet", and this fact reflects their semantic similarity. As the linguist John Firth put it in 1957, "You shall know a word by the company it keeps".

Word2Vec exploits contextual information like this by training a neural net to distinguish actually co-occurring groups of words from randomly grouped words. The input layer takes a sparse representation of a target word together with one or more context words. This input connects to a single, smaller hidden layer.

In one version of the algorithm, the system makes a negative example by substituting a random noise word for the target word. Given the positive example "the plane flies", the system might swap in "jogging" to create the contrasting negative example "the jogging flies".

The other version of the algorithm creates negative examples by pairing the true target word with randomly chosen context words. So it might take the positive examples (the, plane), (flies, plane) and the negative examples (compiled, plane), (who, plane) and learn to identify which pairs actually appeared together in text.

The classifier is not the real goal for either version of the system, however. After the model has been trained, you have an embedding. You can use the weights connecting the input layer with the hidden layer to map sparse representations of words to smaller vectors. This embedding can be reused in other classifiers.

**Training an Embedding as Part of a Larger Model**

You can also learn an embedding as part of the neural network for your target task. This approach gets you an embedding well customized for your particular system, but may take longer than training the embedding separately.

![](https://developers.google.cn/machine-learning/crash-course/images/EmbeddingExample3-1.svg)






