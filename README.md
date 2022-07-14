#Machine Learning Notes

## Linear Regression
### Approximation: 
$$\hat{y} = wx + b$$
### Loss function (mean square error): 
$$J = \frac{1}{N} \sum_{i=1}^n (\vec{w} \cdot \vec{x_i} + b - y_i)^2$$
### Update rules (gradient descent): 
$$w_i = w_i - \frac{2}{N} \sum_{i=1}^n (\vec{w} \cdot \vec{x_i} + b - y_i) \cdot x_i$$ 
</br>

$$b = b - \frac{2}{N} \sum_{i=1}^n (\vec{w} \cdot \vec{x_i} + b - y_i)$$

## Logistic Regression
### Approximation: 
$$\hat{y} = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$
### Linear model (also the decision boundary):
$$f(x) = \vec{w} \cdot \vec{x} + b$$
### Sigmoid function
$$g(x) = \frac{1}{1 + e^{-x}}$$
### Loss function (mean square error): 
$$J = \sum_{i=0}^n y_i \cdot log(\frac{e^{-(\vec{w} \cdot \vec{x} + b)}}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}) + (1 - y_i) \cdot log(\frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}})$$
### Update rules (gradient descent): 
$$w_i = w_i - \frac{2}{N} \sum_{i=1}^n (\frac{1}{1+e^{-(\vec{w} \cdot \vec{x_i} + b)}} - y_i) \cdot x_i$$
</br>

$$b = b - \frac{2}{N} \sum_{i=1}^n (\frac{1}{1 + e^{-(\vec{w} \cdot \vec{x_i} + b)}} - y_i)$$
### Actually, the update rules of logistic regression shares the same form as those of linear regression. But F(x) is different for these two models.
$$w_i = w_i - \frac{2}{N} \sum_{i=1}^n (F(x_i) - y_i) \cdot x_i$$
</r>

$$b = b - \frac{2}{N} \sum_{i=1}^n (F(x_i) - y_i)$$
### Predict
$$Class_{pred}=argmax(P(Y|x_i))$$
