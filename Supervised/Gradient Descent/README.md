# Gradient Descent

## Definition:

Gradient descent (GD) is an iterative first-order optimisation algorithm used to find a local minimum/maximum of a given function. This method is commonly used in machine learning (ML) and deep learning(DL) to minimise a cost/loss function (e.g. in a linear regression). Due to its importance and ease of implementation, this algorithm is usually taught at the beginning of almost all machine learning courses.

## Learning steps:

The goal of the gradient descent algorithm is to minimize the given function (say cost function). To achieve this goal, it performs two steps iteratively:

- 1. Compute the gradient (slope), the first order derivative of the function at that point

- 2. Make a step (move) in the direction opposite to the gradient, opposite direction of slope increase from the current point by alpha times the gradient at that point

## Datasets:

We assume that $w_0 = 4$, function $f(w) = (w - 2)^2 + 1$.

## Reference:

https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21

https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/

