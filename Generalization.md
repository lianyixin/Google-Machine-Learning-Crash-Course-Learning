## Generalization
The ML fine print  
The following three basic assumptions guide generalization:

We draw examples **independently and identically (i.i.d)** at random from the distribution. In other words, examples don't influence each other. (An alternate explanation: i.i.d. is a way of referring to the randomness of variables.)  
The distribution is **stationary**; that is the distribution doesn't change within the data set.  
We draw examples from partitions from the **same distribution**.

In practice, we sometimes violate these assumptions. For example:

Consider a model that chooses ads to display. The i.i.d. assumption would be violated if the model bases its choice of ads, in part, on what ads the user has previously seen.  
Consider a data set that contains retail sales information for a year. User's purchases change seasonally, which would violate stationarity.

**Never train on test data.** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. For example, high accuracy might indicate that test data has leaked into the training set.

For example, consider a model that predicts whether an email is spam, using the subject line, email body, and sender's email address as features. We apportion the data into training and test sets, with an 80-20 split. After training, the model achieves 99% precision on both the training set and the test set. We'd expect a lower precision on the test set, so we take another look at the data and discover that many of the examples in the test set are duplicates of examples in the training set (we neglected to scrub duplicate entries for the same spam email from our input database before splitting the data). We've inadvertently trained on some of our test data, and as a result, we're no longer accurately measuring how well our model generalizes to new data.

## Validation Set
You can greatly reduce your chances of overfitting by partitioning the data set into the three subsets.

![validation set graph]
(https://developers.google.cn/machine-learning/crash-course/images/WorkflowWithValidationSet.svg)

In this improved workflow:

1. Pick the model that does best on the validation set.  
2. Double-check that model against the test set.  

This is a better workflow because it creates fewer exposures to the test set. If possible, it's a good idea to collect more data to "refresh" the test set and validation set. Starting anew is a great reset.