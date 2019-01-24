## Classification

In order to map a logistic regression value to a binary category, you must define a classification threshold (also called the decision threshold).

Part of choosing a threshold is assessing how much you'll suffer for making a mistake. For example, mistakenly labeling a non-spam message as spam is very bad. However, mistakenly labeling a spam message as non-spam is unpleasant, but hardly the end of your job.

The following sections take a closer look at metrics you can use to evaluate a classification model's predictions, **as well as the impact of changing the classification threshold on these predictions**.

A **true positive** is an outcome where the model correctly predicts the positive class. Similarly, a **true negative** is an outcome where the model correctly predicts the negative class.

A **false positive** is an outcome where the model incorrectly predicts the positive class. And a **false negative** is an outcome where the model incorrectly predicts the negative class.

###Accuracy

Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition:

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$

**Accuracy alone doesn't tell the full story when you're working with a class-imbalanced data set**, like this one, where there is a significant disparity between the number of positive and negative labels.

###Precision and Recall

**Precision** attempts to answer the following question: What proportion of positive identifications was actually correct?

$$\text{Precision} = \frac{TP}{TP+FP}$$

**Recall** attempts to answer the following question: What proportion of actual positives was identified correctly?

$$\text{Recall} = \frac{TP}{TP+FN}$$

To fully evaluate the effectiveness of a model, you must examine both precision and recall. Unfortunately, precision and recall are often in tension. **That is, improving precision typically reduces recall and vice versa**. 

###ROC Curve and AUC

**ROC**

An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

1. True Positive Rate   
2. False Positive Rate

True Positive Rate (TPR) is a synonym for recall and is therefore defined as follows:

$$TPR = \frac{TP} {TP + FN}$$

False Positive Rate (FPR) is defined as follows:

$$FPR = \frac{FP} {FP + TN}$$


Classification: ROC Curve and AUC
Estimated Time: 8 minutes
ROC curve
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

True Positive Rate
False Positive Rate
True Positive Rate (TPR) is a synonym for recall and is therefore defined as follows:

False Positive Rate (FPR) is defined as follows:

An ROC curve plots TPR vs. FPR at different classification thresholds. **Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives**. The following figure shows a typical ROC curve.

![roc curve](https://developers.google.cn/machine-learning/crash-course/images/ROCCurve.svg)

**AUC**

AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

**To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient**. Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called AUC.

AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.

![AUC](https://developers.google.cn/machine-learning/crash-course/images/AUCPredictionsRanked.svg)

AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

**Advantage**

AUC is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.  
AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

**Disadvantage**

Scale invariance is not always desirable. For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.

Classification-threshold invariance is not always desirable. In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.

###Prediction Bias

Prediction bias is a quantity that measures how far apart those two averages are. That is:

$$\text{prediction bias} = \text{average of predictions} - \text{average of labels in data set}$$

Possible root causes of prediction bias are:

Incomplete feature set.   
Noisy data set.   
Buggy pipeline.   
Biased training sample.   
Overly strong regularization

A good model will usually have near-zero bias. That said, a low prediction bias does not prove that your model is good. A really terrible model could have a zero prediction bias. For example, a model that just predicts the mean value for all examples would be a bad model, despite having zero bias.

You might be tempted to correct prediction bias by post-processing the learned model—that is, by adding a calibration layer that adjusts your model's output to reduce the prediction bias. For example, if your model has +3% bias, you could add a calibration layer that lowers the mean prediction by 3%. However, adding a calibration layer is a bad idea for the following reasons:

You're fixing the symptom rather than the cause.  
You've built a more brittle system that you must now keep up to date.

**Bucketing**

Logistic regression predicts a value between 0 and 1. However, all labeled examples are either exactly 0 (meaning, for example, "not spam") or exactly 1 (meaning, for example, "spam"). Therefore, when examining prediction bias, you cannot accurately determine the prediction bias based on only one example; you must examine the prediction bias on a "bucket" of examples. 

Why are the predictions so poor for only part of the model? Here are a few possibilities:

The training set doesn't adequately represent certain subsets of the data space.  
Some subsets of the data set are noisier than others.  
The model is overly regularized. (Consider reducing the value of lambda.)
