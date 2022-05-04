# Principal Component Analysis (PCA)

## Definition:

PCA is used in exploratory data analysis and for multidimensionality reduction. The main idea is to project data points onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data’s variation as possible. In other words, using PCA we remove the redundant and highly-correlated data and we keep only the most significant data features for further analysis.

The first principal component is defined as a direction that maximizes variance of the projected data, the second principal component is a direction orthogonal to the first principal component that is the next one to maximize the variance, etc. It can be proved that the principal components are the eigenvectors of the covariance matrix and are computed either by eigendecomposition of the covariance matrix or by the SVD of the data matrix.

Assume we have data consisting of $m$ variables (or features, or attributes, such as age, height, weight, income, etc.) and n observations (or data points, or samples). We form the $m \times n$ ”feature - observation” matrix where variables are listed in the rows and observations in the columns. Some authors and Python prefer ”observation – feature” matrix instead of ”feature – observation” matrix. 
