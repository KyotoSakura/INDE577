# Ensemble methods

**Ensemble methods** are machine learning methods that aggregate the predictions of a group of base learners in order to form a single learning model. For example, imagine that each person in this [video](https://www.youtube.com/watch?v=iOucwX7Z1HU&t=203s) is a trained machine learning model and notice the average of their predictions is a much more accurate prediction that the individual predictions.

## Definition:

### 1. Bagging:

The term **bagging** referes to **b**ootstrap **agg**regating. **Bootstrapping** is a method of inferring results for a population from results found on a collection of smaller random samples of that population, using replacement during the sampling process. In the context of machine learning, a given set of machine learning model is trained respectively on random samples of training data with replacement (see the above figure), then the combined predictions of each model is **aggregated** and used as a single prediction. For regression tasks this would mean taking the average of the set of model prediction, and for classification taking the majority vote.  

Generally speaking, the models we pick for ensembling will be "dumb learners", meaning models that are barely superior to randomly guessing. Individually the models will perform poorly, but collectively will perform well. 

But why is bagging useful?

> "Each model is trained on a different dataset, because they’re bootstrapped. So inevitably, each model will make different mistakes, and have a distinct error and variance. Both the error and variance get reduced in the aggregation step where, literally in the case of Regression, they are averaged out."

The key idea here is the reduction in variance of the model. Again, from the above article:

> "In a Regression task you can calculate actual variance of the prediction compared to the true targets. If the tree produces results that are too far off from its true targets, it has high-variance and therefore, it is overfit."

To illustrate this concept we will once again consider the iris dataset; and compare the performance between a single decision tree model and a bagging classifier of many depth 1 decision trees, referred to as *decision stumps*. Run the following code cell to load the iris dataset and visualize the data. 

### 2. Random Forest：

Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set. Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance.

### 3. Boosting:

With **AdaBoost**, the training algorithm first trains a base classifier and uses it to make predictions on the training set. Then, each of the missclassified training instances is then given a *relative weight*. The next classifier is then trained on the dataset using these relative weights, and so on. 

The idea is that whenever a classifier missclassifies a data point,this data point is then *boosted* to signal difficulty in classification. 

Another popular boosting method is **gradient boosting**. This method works by sequentially adding predictors to an ensemble, each correcting is predecessor. The difference between this method and AdaBoost is that gradient boosting tries to fit the new predictor to the *residual errors* made by the previous predictor.

## Datasets:

- Penguins

The Penguins Dataset contains size measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica. These data were collected from 2007 - 2009 by Dr. Kristen Gorman's team. It consists of 344 rows and 7 columns. The three different species of penguins are Chinstrap, Adélie, and Gentoo penguins.
