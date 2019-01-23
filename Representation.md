## Representation

In traditional programming, the focus is on code. In machine learning projects, the focus shifts to **representation**. That is, one way developers hone a model is by **adding and improving its features**.


**Mapping Raw Data to Features**  
**Feature engineering** means transforming raw data into a feature vector. Expect to spend significant time doing feature engineering.

**Mapping numeric values**  
Integer and floating-point data don't need a special encoding because they can be multiplied by a numeric weight. As suggested in Figure 2, converting the raw integer value 6 to the feature value 6.0 is trivial.

**Mapping categorical values**  
we can instead create a binary vector for each categorical feature in our model that represents values as follows:

For values that apply to the example, set corresponding vector elements to 1.
Set all other elements to 0.  
The length of this vector is equal to the number of elements in the vocabulary. This representation is called a **one-hot encoding** when a single value is 1, and a **multi-hot encoding** when multiple values are 1.

**Sparse Representation**  
Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for street_name. Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very **inefficient representation** in terms of both storage and computation time when processing these vectors. In this situation, a common approach is to use a sparse representation in which only nonzero values are stored. In sparse representations, an independent model weight is still learned for each feature value, as described above.

## Qualities of Good Features

**Avoid rarely used discrete feature values**

**Prefer clear and obvious meanings**

**Don't mix "magic" values with actual data**  
To explicitly mark magic values, create a Boolean feature that indicates whether or not a quality_rating was supplied. Give this Boolean feature a name like is_quality_rating_defined.

In the original feature, replace the magic values as follows:

For variables that take a finite set of values (discrete variables), add a new value to the set and use it to signify that the feature value is missing.  
For continuous variables, ensure missing values do not affect the model **by using the mean value of the feature's data**.

**Account for upstream instability**  
The definition of a feature shouldn't change over time. For example, the following value is useful because the city name probably won't change. (Note that we'll still need to convert a string like "br/sao_paulo" to a one-hot vector.)   
But gathering a value inferred by another model carries additional costs. Perhaps the value "219" currently represents Sao Paulo, but that representation could easily change on a future run of the other model.

## Cleaning Data

**Scaling feature values**  
Scaling means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1). If a feature set consists of only a single feature, then scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, then feature scaling provides the following benefits:

Helps gradient descent converge more quickly.  
Helps avoid the "NaN trap," in which one number in the model becomes a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.  
Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.

**Scaling**

One obvious way to scale numerical data is to linearly map [min value, max value] to a small scale, such as [-1, +1].

Another popular scaling tactic is to calculate the Z score of each value. The Z score relates the number of standard deviations away from the mean. In other words:  
$$scaled value = (value - mean) / stddev.$$

Scaling with Z scores means that most scaled values will be between -3 and +3, but a few values will be a little higher or lower than that range.

**Handling extreme outliers**  
log data or clip or cap

**Binning**    
In the data set, latitude is a floating-point value. However, it doesn't make sense to represent latitude as a floating-point feature in our model. That's because no linear relationship exists between latitude and housing values. For example, houses in latitude 35 are not  more expensive (or less expensive) than houses at latitude 34. And yet, individual latitudes probably are a pretty good predictor of house values.

To make latitude a helpful predictor, let's divide latitudes into "bins" as suggested by the following figure:

![Binning values](https://developers.google.cn/machine-learning/crash-course/images/ScalingBinningPart2.svg)

Instead of having one floating-point feature, we now have 11 distinct boolean features (LatitudeBin1, LatitudeBin2, ..., LatitudeBin11). Having 11 separate features is somewhat inelegant, so let's unite them into a single 11-element vector. Doing so will enable us to represent latitude 37.4 as follows:[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0].

For simplicity's sake in the latitude example, we used whole numbers as bin boundaries. Had we wanted finer-grain resolution, we could have split bin boundaries at, say, every tenth of a degree. Adding more bins enables the model to learn different behaviors from latitude 37.4 than latitude 37.5, but only if there are sufficient examples at each tenth of a latitude.

Another approach is to bin by quantile, which ensures that the number of examples in each bucket is equal. Binning by quantile completely removes the need to worry about outliers.

**Scrubbing**

Until now, we've assumed that all the data used for training and testing was trustworthy. In real-life, many examples in data sets are unreliable due to one or more of the following:

**Omitted values**. For instance, a person forgot to enter a value for a house's age.  
**Duplicate examples**. For example, a server mistakenly uploaded the same logs twice.  
**Bad labels**. For instance, a person mislabeled a picture of an oak tree as a maple.  
**Bad feature values**. For example, someone typed in an extra digit, or a thermometer was left out in the sun.

Once detected, you typically "fix" bad examples by removing them from the data set. To detect omitted values or duplicated examples, **you can write a simple program**. Detecting bad feature values or labels can be far trickier.

In addition to detecting bad individual examples, you must also detect bad data in the aggregate. Histograms are a great mechanism for visualizing your data in the aggregate.

**Know your data**  
Follow these rules:

Keep in mind what you think your data should look like.  
Verify that the data meets these expectations (or that you can explain why it doesn’t).  
Double-check that the training data agrees with other sources (for example, dashboards).
