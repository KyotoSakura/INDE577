# The Single Neuron Linear Regression Model

In this notebook we implement the single neuron model together with the gradient descent algorithm in order to solve the **linear regression problem**. 

## Regression

Let $\mathcal{X}$ be the space of all possible feature vectors, let $\mathcal{Y}$ be the space of all possible corresponding labels for the feature vectors, and let $f:\mathcal{X} \rightarrow \mathcal{Y}$ be the optimal target function assigning labels to feature vectors in $\mathcal{Y}$. Next recall that in supervised machine learning we observe some subset of features and labels as shown in the figure below. 

In [regression](https://favtutor.com/blogs/types-of-regression), machine learning models are given labeled data $\mathcal{D} = \{(\mathbf{x}^1, y^1), \dots, (\mathbf{x}^N, y^N)\}$, where the feature vectors satisfy $\mathbf{x}^{(i)} \in \mathbb{R}$ and the target labels satify $y^{(i)} \in \mathbb{R}$. Thus, this supervised learning task seeks to predict real valued target labels. This is different from classification (such as the perceptron single neuron model) as the following figure suggests.

### - Linear Regression

In this notebook we will focus on **linear regression**. This specific case of regression assumes that the *target values in $\mathcal{Y}$ are approximated by a linear function of the associated feature vectors*. That is, the optimal target function $f:\mathcal{X} \rightarrow \mathcal{Y}$ is assumed the be roughly a linear function. 

General ML Model:
---

Because we are assuming the target function $f:\mathcal{X} \rightarrow \mathcal{Y}$ is a **linear function of the input features**, and because we know single neuron models are good function approximators, we next build a single neuron model with a *linear-activation* activation function. Furthermore, in this model we choose the *mean-sqaured error* cost function:

$$
C(\mathbf{w}, b) = \frac{1}{2N}\sum_{i=1}^{N}\Big(\hat{y}^{(i)} - y^{(i)}\Big)^2. 
$$

With our specific case of linear regression on the setosa iris dataset with a single feature measurment taken from sepal length data, we next construct a single neuron model with a linear activation function and the mean-sqaured error cost function as depicted in the figure below.

## Minimize the Cost Function $C(w_1, b)$

Before defining a custom ```SingleNeuron``` class, we first need first discuss how to minimize the neurons cost function. More specifically, we wish to solve the following optimization problem:

$$
\min_{w_1, b}C(w_1, b)
$$

Since $C(w_1, b)$ is a differentiable function of both $w_1$ and $b$, we may attempt to solve this minimization problem by applying the gradient descent algorithm:

$$
w_1 \leftarrow w_1 - \alpha \frac{\partial C}{\partial w_1}
$$

$$
b \leftarrow b - \alpha \frac{\partial C}{\partial b}
$$

### - Finding the Partial Derivatives of $C(w_1, b)$ 
In order to implement the gradient descent method we first need to understand how the partial derivatives of $C(w_1, b)$ are calculated over the training data at hand. With this in mind, suppose for now that we are calculating the mean-sqaured error cost function on a *single example* example of data, i.e., $N = 1$. For this single example we observe that the mean-sqaured error cost function becomes: 

$$
C(w, b; \mathbf{x}^{(i)}, y^{(i)}) = \frac{1}{2}\Big(\hat{y}^{(i)} - y^{(i)}\Big)^2. 
$$

In the case of a linear activation function, it is important to note that $\hat{y}^{(i)}$ is a very simple function of both $w_1$ and $b$. More specifically, we observe:

$$
\hat{y}^{(i)} = a = z = w_1x^{(i)} + b. 
$$

Thus, we may rewrite our neuron cost function with a single observation:

$$
C(w, b; \mathbf{x}^{(i)}, y^{(i)}) = \frac{1}{2}\Big(w_1x^{(i)} + b - y^{(i)}\Big)^2. 
$$

With this equation, we can calculate $\partial C/ \partial w_1$ and $\partial C/ \partial b$ easily by applying the [chain rule (click for a quick refresher on the concept)](https://www.youtube.com/watch?v=HaHsqDjWMLU). The resulting partial derivatives with respect to $w_1$ and $b$ shown by the following equations:

1. $\frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1} = (w_1x^{(i)} + b - y^{(i)})x^{(i)} = (\hat{y}^{(i)} - y^{(i)})x^{(i)}$
2. $\frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial b} = (w_1x^{(i)} + b - y^{(i)}) = (\hat{y}^{(i)} - y^{(i)})$

Understanding the different ways in which we may calculate the partial derivatives of our cost function is essential in applying any *first-order* minimization technique on the cost function $C(w_1, b)$. With what follows we discuss two of the three fundamental methods used to accomplish this goal. 


### - Different Flavors of First-Order Minimization 
When considering a single instance of data, we easily calculated $\frac{\partial C}{\partial w_1}$ and $\frac{\partial C}{\partial b}$ by applying the chain-rule. This notion can now be extended to all data used in training by summing the gradients calculated at entry of data. We will refer to this process as calculating the **full gradient** (or **full partial derivatives**) with respect to the training data: 

1. $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1} = \frac{1}{N}\sum_{i=1}^{N}\Big(\hat{y}^{(i)} - y^{(i)}\Big)x^{(i)}$
2. $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}\Big(\hat{y}^{(i)} - y^{(i)}\Big)$

Calculating the full gradient with respect to all training data and applying the gradient descent algorithm is called **batch gradient descent**.

**Flavor 1. Batch Gradient Descent Algorithm:**
1. For each epoch **do**
2. Calculate the full gradient by finding $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1}$ and $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b}$.
3. $w \leftarrow w - \alpha \frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1}$
4. $b \leftarrow b - \alpha \frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b}$

Applying batch gradient descent will work. However, *this method can be very slow and use a lot of memory*, especially when the number of training data is very large (possibly millions). More importantly, **batch gradient descent is not necessary to find local minima**. 

The most common way work around for this problem is to update $w_1$ and $b$ by calculating the gradient with respect to one entry of data at a time. This technique is called **stochastic gradient descent** and is one of the primary tools in training deep neural networks and simple single neuron models.  

**Flavor 2. Stochastic Gradient Descent Algorithm:**
1. For each epoch **do**
2. For $i = 1, \dots, N$ **do**
3. Calculate $\frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1}$ and $\frac{\partial C(\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)}))}{\partial b}$.
2. $w \leftarrow w - \alpha \frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1}$
3. $b \leftarrow b - \alpha \frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial b}$

For single neuron models in practice, stochastic gradient descent should be the preferred way for optimizing the weights and bias by minimizing the cost function. We implement stochastic gradient descent with the ```train``` method used in the following custom ```SingleNeuron``` class. 

# Datasets:

- Penguins

The Penguins Dataset contains size measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica. These data were collected from 2007 - 2009 by Dr. Kristen Gorman's team. It consists of 344 rows and 7 columns. The three different species of penguins are Chinstrap, Adélie, and Gentoo penguins.


