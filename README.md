# SharpEye
SharpEye is a collection of MATLAB functions for the logistic regression algorithm and its application has been implemented for binary image classification.

This project trains a logistic regression model as an example, to predict if the person in a picture (such as _Elon Musk!_) is wearing a mask (_musk!_) or not. 

The datasets' raw pictures were gathered from [here](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset) and they were converted to `.h5` files using [this](https://github.com/rezmansouri/SharpEye/blob/main/assets/create_h5_dataset.py) python script.

## Overview
### Logistic Regression
This type of statistical model (also known as logit model) is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. In logistic regression, a logit transformation (sigmoid function) is applied on the odds—that is, the probability of success divided by the probability of failure.
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/fig1.png"/>
  </br>
  <i>Figure 1 - Logistic Regression</i>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\sigma&space;(z)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}"/>
  </br>
  <i>Sigmoid Function</i>
</p>

### Gradient Descent
Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. If _w_ is a weight vector of our model, we subtract _dw_ times a learning rate _a_ from _w_ in each backpropagation. This means: _w = w - a * dw_.
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/fig2.png"/>
  </br>
  <i>Figure 2 - Gradient Descent</i>
</p>

### Explanation
We have a model with input dimension of **n** and output dimension of **1**. For each of these **n** inputs, there is a weight associeted with in the model, and an overall bias. Our input is image data. Each image is of 64 x 64 x 3 dimensions. In order to convert each image to a feature vector, we flatten it to a 12288 x 1 vector. Now suppose our training data has 1000 images. Thus, the overall input would be of 12288 x 1000 dimensions. Lets call this input _X_. This type of preprocessing, allows vectorized implementation, which is much faster than while loops for each of the images. In order to account for each of the elements in a column (single image) of _X_, a weight is considered. Accordingly, our weights matrix, _W_'s dimensions would be of 12288 x 1. Finally the last parameter _b_ is a single real number.  So far the parameters are as follows:
- _W_ : the weight matrix of dimension (12288, 1)
- _b_ : the bias of the model, a single real number

The above parameters are learnable, meaning they will be optimized as the gradient descent converges. Initially, they will be assigned small values like 0.02.
- _X_ : the input matrix of dimension (12288, 1000)

#### Forward Propagation
The forward propagation, calculates the model's predictions on the input data. The calculation is done in two parts.
##### Linear Forward Prop.
The linear part is to calculate the dot product of _X_ and _W_<sup><i>T</i></sup> plus _b_. Lets call this _Z_. The elements in _Z_ could be too large or too small, making it difficult to map these values for a prediction scale on say, 0 to 1. Thus, the Non-Linear part is required which is the heart of this model, ***Logistic Regression***!
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?Z&space;=&space;X&space;\bullet&space;W^{T}&space;&plus;&space;b"/>
  </br>
  <i>Linear Part of Forward Prop.</i>
</p>

##### Non-Linear Forward Prop
Pass _Z_ to the sigmoid function. Lets call the output _A_ (activations). It can be seen that the dimension of _A_ would be 1 x 1000. This will result in values between zero and one. Now we can put a 0.5 threshold on them and say the values below this are considered as zero (false) and above this, as one (true) and call that _Y_<sub><i>hat</i></sub>. _Remember that we are doing binary classification_. Each element in this vector corresponds to our prediction from each image (column) in _X_.
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?A&space;=&space;\sigma&space;(Z)"/>
  </br>
  <i>Non-Linear Part of Forward Prop.</i>
</p>

#### Backward Propagation
In backward propagation, we calculate the logistic cost of our predictions, according to their true labels _Y_ (also a 1 x 1000 vector). The formula for logistic cost is as follows:
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?J&space;=&space;-\frac{1}{m}[(Y&space;*&space;log(A))&space;&plus;&space;((1-Y)&space;*&space;log(1-A))]"/>
  </br>
  <i>Logistic Cost</i>
</p>

Where **m** is the number of input samples, in our case, 1000. _Note that the operator <b>*</b> is element-wise multiplication._

After calculating this value, the derivatives of _W_ and _b_ with respect to the cost, (_J_) are calculated. We'll call them _dW_ and _db_.
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?dW&space;=&space;\frac{1}{m}X&space;*&space;(A&space;-&space;Y)^{T}"/>
  </br>
  <i>Derivative of W with respect to the Logistic Cost - a vector with the same dimensions as W</i>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?dW&space;=&space;\frac{1}{m}\sum&space;(A&space;-&space;Y)"/>
  </br>
  <i>Derivative of b with respect to the Logistic Cost - a single real number, just like b</i>
</p>

Note that although the dimensions of _X_ and _(A - Y)_<sup><i>T</i></sup> do not match, the element-wise product is caluclatable through repeating the multiplication of _(A - Y)_<sup><i>T</i></sup> on each column of _X_. This is referred to _Broadcasting_ in the python's numpy. You can read more [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Now that we have _dW_ and _db_, we perform on operation of gradient descent. Using the learning rate hyperparameter (_a_) of something like 0.01, _W_ and _b_ will change their values:
- _W = W - a * dW_
- _b = b - a * db_

### Conclusion
After repeating forward / backward propagation for a specific number of iterations, say 1000, the model is ready to be tested and if approved, put in use for making predictions!