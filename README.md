# Machine Learning Laboratory

This repository contains implementations of fundamental machine learning algorithms from scratch and using standard libraries. Each algorithm includes explanations, sample code, and important viva questions.

## Algorithms Implemented

### 1. Find-S Algorithm
**Concept:** 
- Finds the most specific hypothesis consistent with training examples
- Used in concept learning problems


### 2. Candidate Elimination Algorithm
**Concept:**
- Maintains version space (set of all consistent hypotheses)
- Updates both general and specific boundaries



### 3. Decision Tree (ID3)
**Concept:**
- Uses information gain (entropy reduction) for splitting
- Recursively builds tree until pure leaves or stopping criteria



### 4. Naive Bayesian Classifier
**Concept:**
- Applies Bayes' theorem with "naive" feature independence assumption
- Works well for text classification and other high-dimensional data



### 5. K-Means Clustering
**Concept:**
- Unsupervised learning algorithm for clustering
- Iteratively assigns points to nearest centroid and updates centroids



### 6. K-Nearest Neighbors (KNN)
**Concept:**
- Instance-based learning where classification is based on majority vote of neighbors
- Requires distance metric (typically Euclidean)


### 7. Backpropagation Algorithm
**Concept:**
- Training algorithm for artificial neural networks
- Uses gradient descent to minimize error by adjusting weights



### 8. Locally Weighted Regression
**Concept:**
- Non-parametric regression method that gives higher weight to nearby points
- Useful when global parametric models don't fit well



## Viva Questions (50-100)

### Fundamental Concepts
1. **Q:** What is the difference between supervised and unsupervised learning?  
   **A:** Supervised uses labeled data to predict outcomes, unsupervised finds patterns in unlabeled data.

2. **Q:** Explain bias-variance tradeoff.  
   **A:** Simpler models have high bias but low variance, complex models have low bias but high variance.

### Find-S & Candidate Elimination
3. **Q:** What is the main limitation of Find-S algorithm?  
   **A:** It only finds the most specific hypothesis and may miss others consistent with data.

4. **Q:** How does candidate elimination maintain version space?  
   **A:** By keeping track of most general (G) and most specific (S) boundaries.

### Decision Trees
5. **Q:** Why is information gain used in ID3?  
   **A:** It measures how well a feature splits the data to reduce uncertainty (entropy).

6. **Q:** How does pruning help decision trees?  
   **A:** It prevents overfitting by removing branches with little predictive power.

### Naive Bayes
7. **Q:** Why is it called "naive"?  
   **A:** Because it assumes all features are independent given the class, which is rarely true.

8. **Q:** When does Naive Bayes perform well despite its assumption?  
   **A:** In text classification where feature independence isn't crucial for good performance.

### K-Means
9. **Q:** How do you choose K in K-means?  
   **A:** Using elbow method (where WCSS stops decreasing significantly) or domain knowledge.

10. **Q:** What are limitations of K-means?  
    **A:** Sensitive to initial centroids, assumes spherical clusters of similar size.

### KNN
11. **Q:** How does choice of K affect KNN performance?  
    **A:** Small K leads to noisy decision boundaries, large K makes boundaries smoother but may miss local patterns.

12. **Q:** Why is feature scaling important in KNN?  
    **A:** Because it uses distance metrics where features on larger scales dominate.

### Backpropagation
13. **Q:** What is vanishing gradient problem?  
    **A:** When gradients become extremely small during backpropagation, slowing learning in early layers.

14. **Q:** Why use activation functions in neural networks?  
    **A:** To introduce non-linearity so the network can learn complex patterns.

### Regression
15. **Q:** When would you use locally weighted regression?  
    **A:** When relationship between variables changes in different regions of feature space.

16. **Q:** How does locally weighted regression differ from standard linear regression?  
    **A:** It fits a new model for each prediction point, weighting nearby points more.
