# SharpEye 🧿
SharpEye is a collection of MATLAB functions for the logistic regression algorithm and its application has been implemented for binary image classification.
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/figures/header.png"/>
  </br>
</p>

This project trains a logistic regression model as an example, to predict if the person in a picture (such as _Elon Musk!_) is wearing a mask (_musk!_) or not. 

The datasets' raw pictures were gathered from [here](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset) and they were converted to `.h5` files using [this](https://github.com/rezmansouri/SharpEye/blob/main/assets/create_h5_dataset.py) python script.

## Overview
### Logistic Regression
This type of statistical model (also known as logit model) is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. In logistic regression, a logit transformation (sigmoid function) is applied on the odds—that is, the probability of success divided by the probability of failure.
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/figures/fig1.png"/>
  </br>
  <i>Figure 1 - Logistic Regression</i>
  </br>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\sigma&space;(z)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}"/>
  </br>
  <i>Sigmoid Function</i>
</p>

### Gradient Descent
Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. If _w_ is a weight vector of our model, we subtract _dw_ times a learning rate _a_ from _w_ in each backpropagation. This means: _w = w - a * dw_.
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/figures/fig2.png"/>
  </br>
  <i>Figure 2 - Gradient Descent</i>
</p>

### Explanation
We have a model with input dimension of **n** and output dimension of **1**. For each of these **n** inputs, there is a weight associeted with in the model, and an overall bias. Our input is image data. Each image is of 64 x 64 x 3 dimensions. In order to convert each image to a feature vector, we flatten it to a 12288 x 1 vector. Now suppose our training data has 1000 images. Thus, the overall input would be of 12288 x 1000 dimensions. Let's call this input _X_. This type of preprocessing, allows vectorized implementation, which is much faster than while loops for each of the images. In order to account for each of the elements in a column (single image) of _X_, a weight is considered. Accordingly, our weights matrix, _W_'s dimensions would be of 12288 x 1. Finally the last parameter _b_ is a single real number.  So far the parameters are as follows:
- _W_ : the weight matrix of dimension (12288, 1)
- _b_ : the bias of the model, a single real number

The above parameters are learnable, meaning they will be optimized as the gradient descent converges. Initially, they will be assigned small values like 0.02.
- _X_ : the input matrix of dimension (12288, 1000)

#### Forward Propagation
The forward propagation, calculates the model's predictions on the input data. The calculation is done in two parts.
##### Linear Forward Prop.
The linear part is to calculate the dot product of _W_<sup><i>T</i></sup> and _X_ plus _b_. Let's call this _Z_. The elements in _Z_ could be too large or too small, making it difficult to map these values for a prediction scale on say, 0 to 1. Thus, the Non-Linear part is required which is the heart of this model, ***Logistic Regression***!
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?Z&space;=&space;W^{T}&space;\bullet&space;X&space;&plus;&space;b"/>
  </br>
  <i>Linear Part of Forward Prop.</i>
</p>

##### Non-Linear Forward Prop
Pass _Z_ to the sigmoid function. Let's call the output _A_ (activations). It can be seen that the dimension of _A_ would be 1 x 1000. This will result in values between zero and one. Now we can put a 0.5 threshold on them and say the values below this are considered as zero (false) and above this, as one (true) and call that _Y_<sub><i>hat</i></sub>. _Remember that we are doing binary classification_. Each element in this vector corresponds to our prediction from each image (column) in _X_.
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?A&space;=&space;\sigma&space;(Z)"/>
  </br>
  <i>Non-Linear Part of Forward Prop.</i>
</p>

Figure 3 describes the model and forward prop.
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/figures/fig3.png"/>
  </br>
  <i>Figure 3 - Our Model</i>
</p>

#### Backward Propagation
In backward propagation, we calculate the logistic cost of our predictions, according to their true labels _Y_ (also a 1 x 1000 vector). The formula for logistic cost is as follows:
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?J&space;=&space;-\frac{1}{m}\sum[(Y&space;*&space;log(A))&space;&plus;&space;((1-Y)&space;*&space;log(1-A))]"/>
  </br>
  <i>Logistic Cost</i>
</p>

Where **m** is the number of input samples, in our case, 1000. _Note that the operator <b>*</b> is element-wise multiplication._

After calculating this value, the derivatives of the cost, (_J_), with respect to _W_ and _b_  are calculated. We'll call them _dW_ and _db_.
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;J}{\partial&space;W}&space;=&space;dW&space;=&space;\frac{1}{m}&space;X&space;*&space;(A-Y)^{T}"/>
  </br>
  <i>Derivative of J, the Logistic Cost, with respect to W - a vector with the same dimensions as W</i>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;J}{\partial&space;b}&space;=&space;db&space;=&space;\frac{1}{m}\sum(A-Y)"/>
  </br>
  <i>Derivative of J, the Logistic Cost, with respect to b - a single real number, just like b</i>
</p>

Note that although the dimensions of _X_ and _(A - Y)_<sup><i>T</i></sup> do not match, the element-wise product is caluclatable through repeating the multiplication of _(A - Y)_<sup><i>T</i></sup> on each column of _X_. This is referred to as _Broadcasting_ in python's numpy. You can read more [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Now that we have _dW_ and _db_, we perform an operation of gradient descent. Using the learning rate hyperparameter (_a_) of something like 0.01, _W_ and _b_ will change their values:
- _W = W - a * dW_
- _b = b - a * db_

### Conclusion
After repeating forward / backward propagation for a specific number of iterations, say 1000, the model is ready to be tested and if approved, put in use for making predictions!

## Scenario
Here begins the MATLAB part of the story (_Yuck! MATLAB is so dumb_). This scenario is the `main.m` script which can be found [here](https://github.com/rezmansouri/SharpEye/blob/main/SharpEye/main.m). I will explain each section of it here.

First we need to create an object of our _SharpEye_ function collection.
```
eye = SharpEye;
```

Then we read our datasets. 
```
x_train_original = eye.read_h5_correctly('../datasets/train_maskvnomask.h5', '/train_set_x');
y_train_original = eye.read_h5_correctly('../datasets/train_maskvnomask.h5', '/train_set_y');
x_test_original = eye.read_h5_correctly('../datasets/test_maskvnomask.h5', '/test_set_x');
y_test_original = eye.read_h5_correctly('../datasets/test_maskvnomask.h5', '/test_set_y');
```
It should be noted that MATLAB reads `.h5` files wrong and results in transposed values. Thus, I had to implement my own read function (`read_h5_correctly`) which I'll explain in the next section.

After, I get the size of the train/test datasets to have their numbers of samples and picture dimensions. Although in the overview, our example had 64 x 64 pictures, the implementation allows training with dynamic model dimensions resulting from the dimensions of the input.
```
x_train_original_size = size(x_train_original);
x_test_original_size = size(x_test_original);

m_train = x_train_original_size(1);
m_test = x_test_original_size(1);
num_px = x_train_original_size(2);
```
Then, as explained in the overview, I flatten the datasets, while casting their type from int8 to double for the sake of computations.
```
x_train_flat = double(reshape(x_train_original, m_train, [], 1)');
x_test_flat = double(reshape(x_test_original, m_test, [], 1)');
y_train = double(y_train_original);
y_test = double(y_test_original);
```
Now it's time to normalize the input. RGB image pixel values is between 0 and 255. By dividing pixel values to 255, they will be normalized. This is a common practice among the computer vision communities.
```
x_train = x_train_flat / 255;
x_test = x_test_flat / 255;
```
Here I create the required variables for the model to be generated. In `get_initial_parameters` function, which I'll explain later, the weight/bias initialization is done.
```
model_dimension = [num_px * num_px * 3, 1];
[weights, bias] = eye.get_initial_parameters(model_dimension);
```
Now the model is ready to be generated and trained.
```
[final_weights, final_bias, costs] = eye.optimize(weights, bias, x_train, y_train, 1000, 0.001);
```
The last two arguments are the number of iterations (epochs) and the learninig rate, respectively.

Now that the model is ready, it's time to see how it did.
```
train_acc = eye.predict(final_weights, final_bias, x_train, y_train);
fprintf('train accuracy: %f\n', train_acc);
test_acc = eye.predict(final_weights, final_bias, x_test, y_test);
fprintf('test accuracy: %f\n', test_acc);
```
The output of these last two steps would be:
```
epoch 1000 / 1000
final cost: 0.224652
train accuracy: 92.819979
test accuracy: 92.194093
```
And the consecutive costs of training are also plotted:
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/figures/fig4.jpg"/>
  </br>
  <i>Figure 4 - Plot of the costs</i>
</p>
It can be seen that it keeps going down smoothely, so we are doing just right!

Now you can give 64 * 64 (or whatever dimension your training/test dataset images have) to the _SharpEye_ model and let it spit out the prediction.

In our case, we were trying to find if people are wearing masks or not. So let's see if our model can figure out if elon musk is wearing a _musk!_ or not.
```
eye.tell('../images/elon-with-mask.jpg', final_weights, final_bias, num_px);
eye.tell('../images/elon-without-mask.jpg', final_weights, final_bias, num_px);
```
<p align="center">
  <img src="https://github.com/rezmansouri/SharpEye/blob/main/assets/figures/fig5.png"/>
  </br>
</p>

Looks like elon can be caught if he is not wearing his _Musk!_

## SharpEye Functions
This sections explains the implementation of the functions used in the scenario above. The functions are in [this](https://github.com/rezmansouri/SharpEye/blob/main/SharpEye/SharpEye.m) file.
1. ### read_h5_correctly

In this function, first using the default h5d5 MATLAB package, the dataset is read. Then, using `permute` it is turned back to its original demensions (_detranspose!_)
```
function data = read_h5_correctly(address, dataset_name)
            incorrect = h5read(address, dataset_name);
            ndim = numel(size(incorrect));
            data = permute(incorrect, ndim:-1:1);
end
```
2. ### get_initial_parameters

First we set the range to 1 (equivalent for random.seed() in python's numpy). A common practice where you set the random seed to a constant so that with each execution, the same random numbers are generated as before. Then we initialize the weights with small values and the bias with zero. This is a newbie type of initialization. Normally you would want to initialize the parameters using a _He Initialization_ approach which leads to faster gradient descent convergance. But this does its job for us.
```
function [weights, bias] = get_initial_parameters(dimension)
            rng(1);
            weights = rand(dimension) * 0.01;
            bias = .0;
end
```
3. ### sigmoid
The default matlab R2015b does not have the sigmoid function, so I had to implement it myself.
```
function result = sigmoid(input)
            result = 1 ./ (1 + exp(-input));
end
```
4. ### forward_propagation
We calculate the dot product of _W_<sup><i>T</i></sup> and _X_ plus _b_ and take the sigmoid of it, and return in as the _A_ vector.
```
function activations = forward_propagation(w, b, x)
            z = w' * x + b;
            activations = SharpEye.sigmoid(z);
end
```
5. ### compute_cost
Calculating the logistic cost function for one forward propagation.
```
function cost = compute_cost(activations, y, m)
            cost_part_one = y .* log(activations);
            cost_part_two = (1 - y) .* log(1 - activations);
            cost = -sum(cost_part_one + cost_part_two) / m;
            cost = squeeze(cost);
end
```
6. ### backward_propagation
Calculating the derivatives of the cost _J_ with respect to _W_ and _b_.
```
function [dw, db] = backward_propagation(x, y, activations, m)
            dw = x * (activations - y)' / m;
            db = sum(activations - y) / m;
end
```
7. ### optimize
This is the function that repeats forward/back prop for `num_iterations` numbers. It uses the functions described above. At the end it plots the costs of training, prints the final cost, and returns the final_weights, final_biases and the costs.
```
function [final_weights, final_biases, costs] = optimize(w, b, x, y, num_iterations, learning_rate)
    costs = zeros([1, num_iterations]);
    x_size = size(x);
    for i=1:num_iterations
        activations = SharpEye.forward_propagation(w, b, x);
        cost = SharpEye.compute_cost(activations, y, x_size(2));
        [dw, db] = SharpEye.backward_propagation(x, y, activations, x_size(2));
        w = w - dw * learning_rate;
        b = b - db * learning_rate;
        costs(i) = cost;
        if mod(i, 10) == 0
            clc;
            fprintf('epoch %d / %d\n', i, num_iterations);
        end
    end
    final_weights = w;
    final_biases = b;
    plot(costs)
    fprintf('final cost: %f\n', costs(end));
end
```
8. ### predict
This function performs forward prop on a set of inputs and checks with their true labels to calculate a accuracy index. It should be noted that where `y_hat(activations >= 0.5) = 1;` is done, an array of zero values with the size of `activations` is manuplated to contain ones where the corresponding elements in the `activations` array are more or equal to 0.5. Basically applying the threshold to _A_ in order to make _Y_<sub><i>hat</i></sub>.
```
function accuracy = predict(w, b, x, y)
            x_size = size(x);
            m = x_size(2);
            activations = SharpEye.forward_propagation(w, b, x);
            y_hat = zeros(size(activations));
            y_hat(activations >= 0.5) = 1;
            accuracy = numel(find(y_hat==y)) / m * 100;
end
```

9. ### tell
Creating _X_ from a single image (with matching dimensions with the training/test datasets) and performing one forward prop. to make predictions about that image.
```
function tell(picture_address, w, b, num_px)
    pic = imread(picture_address);
    pic_size = size(pic);
    if pic_size(1) ~= num_px || pic_size(2) ~= num_px
        fprintf('Incorrect image dimensions, image should be %dx%d', num_px, num_px);
    end
    x_flat = double(reshape(pic, 1, [], 1)');
    x = x_flat / 255;
    y = SharpEye.forward_propagation(w, b, x);
    y = floor(y(1) + 0.5);
    figure;
    imshow(pic);
    if y == 1
        title('wearing a mask');
    else
        title('NOT wearing a mask');
    end
end
```

## Contributions
This project was a starting point for me in deep learning and computer vision. Other than educational purposes, I don't think it would have any other use. However who am I to reject pull requests?! Please do consider doing so if you wish to. 
