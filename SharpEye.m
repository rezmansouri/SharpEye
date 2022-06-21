classdef SharpEye
    methods (Static)
%% This function is implemented to read h5d5 datasets correctly, matlab does it according to Fortran's approach in reading files, this way, it is done the way C family languages do. Otherwise, the datasets would be transposed!
        function data = read_h5_correctly(address, dataset_name)
            incorrect = h5read(address, dataset_name);
            ndim = numel(size(incorrect));
            data = permute(incorrect, ndim:-1:1);
        end
%% To Initialize the learnable parameters. The weights matrix is a dimension * 1 vector and bias is a single floating point.
        function [weights, bias] = get_initial_parameters(dimension)
            rng(1);
            weights = rand(dimension) * 0.01;
            bias = .0;
        end
%% The sigmoid function, to map the linear computations to the range of 0 < x < 1.
        function result = sigmoid(input)
            result = 1 ./ (1 + exp(-input));
        end
%% Forward prop: computing the linear matrix multiplication of the input and w, and summing with b, and finally applying the sigmoid function. Further, computing the logistic regression cost.
        function activations = forward_propagation(w, b, x)
            z = w' * x + b;
            activations = SharpEye.sigmoid(z);
        end
%% Binary classification (Logistic Regression) cost.
        function cost = compute_cost(activations, y, m)
            cost_part_one = y .* log(activations);
            cost_part_two = (1 - y) .* log(1 - activations);
            cost = -sum(cost_part_one + cost_part_two) / m;
            cost = squeeze(cost);
        end
%% Backward prop: computing the derivatives of w and b with respect to the cost function.
        function [dw, db] = backward_propagation(x, y, activations, m)
            dw = x * (activations - y)' / m;
            db = sum(activations - y) / m;
        end
%% The heart of the algorithm, doing forward prop, backward prop, and updating the parameters for a predefined number of epochs.
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
%% The forward prop process applied to a dataset, followed by rounding up the predictions of > 0.5 as MASK and < 0.5 as NO MASK.
        function accuracy = predict(w, b, x, y)
            x_size = size(x);
            m = x_size(2);
            activations = SharpEye.forward_propagation(w, b, x);
            y_hat = zeros(size(activations));
            y_hat(activations >= 0.5) = 1;
            accuracy = numel(find(y_hat==y)) / m * 100;
        end
%% The function to make predictions on a single picture.
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
                title('this person is wearing a mask');
            else
                title('this person is NOT wearing a mask');
            end
        end
%%
    end
end