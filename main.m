clear;
clc;

eye = SharpEye;

x_train_original = eye.read_h5_correctly('./datasets/train_maskvnomask.h5', '/train_set_x');
y_train_original = eye.read_h5_correctly('./datasets/train_maskvnomask.h5', '/train_set_y');
x_test_original = eye.read_h5_correctly('./datasets/test_maskvnomask.h5', '/test_set_x');
y_test_original = eye.read_h5_correctly('./datasets/test_maskvnomask.h5', '/test_set_y');
list_classes = eye.read_h5_correctly('./datasets/test_maskvnomask.h5', '/list_classes');

x_train_original_size = size(x_train_original);
x_test_original_size = size(x_test_original);

m_train = x_train_original_size(1);
m_test = x_test_original_size(1);
num_px = x_train_original_size(2);

x_train_flat = double(reshape(x_train_original, m_train, [], 1)');
x_test_flat = double(reshape(x_test_original, m_test, [], 1)');

x_train = x_train_flat / 255;
x_test = x_test_flat / 255;

y_train = double(y_train_original);
y_test = double(y_test_original);

model_dimension = [num_px * num_px * 3, 1];
[weights, bias] = eye.get_initial_parameters(model_dimension);

[final_weights, final_bias, costs] = eye.optimize(weights, bias, x_train, y_train, 1000, 0.001);

train_acc = eye.predict(final_weights, final_bias, x_train, y_train);
fprintf('train accuracy: %f\n', train_acc);
test_acc = eye.predict(final_weights, final_bias, x_test, y_test);
fprintf('test accuracy: %f\n', test_acc);

eye.tell('images/elon-with-mask.jpg', final_weights, final_bias, num_px);
eye.tell('images/elon-without-mask.jpg', final_weights, final_bias, num_px);