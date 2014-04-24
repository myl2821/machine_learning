function [J grad] = nnCostFunction(nn_params, ... %[Theta1(:) ; Theta2(:)];
                                   input_layer_size, ... %400
                                   hidden_layer_size, ... %25
                                   num_labels, ... %10
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));%25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));%10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

error = zeros(m,num_labels);

%-----------------------feedForward--------
A1 = [ones(m,1) X]; % 5000 x 401
Z2 = A1 * Theta1';% 5000 x 25
A2 = [ones(size(Z2,1),1) sigmoid(Z2)]; % 5000 x 26
Z2 = [ones(size(Z2,1),1) Z2];
Z3 = A2 * Theta2'; % 5000 x 10
A3 = sigmoid(Z3);% 5000 x 10

yy = zeros(m,num_labels);

for i = 1:m
    yy(i,y(i)) = 1;
end

for i = 1:m
J = J + (-log(A3(i,:)) * yy(i,:)' - log(1-A3(i,:)) * (1-yy(i,:)'));
error(i,:) = A3(i,:) - yy(i,:);
end

J = 1/m * J; 

T1 = Theta1; 
T1(:,1) = [];
T2 = Theta2; 
T2(:,1) = [];
T1 = T1.^2;
T2 = T2.^2;

J_reg = lambda / (2*m) * (sum(sum(T1)) + sum(sum(T2)));
J = J + J_reg;


%----------------bp---------------------
delta3 = A3 - yy; % 5000 x 10


for i = 1:m
d3 = delta3(i,:)';% 10 x 1
d2 = (Theta2' * d3) .* sigmoidGradient(Z2(i,:))'; % 26 x 1
d2 = d2(2:end);
Theta2_grad = Theta2_grad + d3 * (A2(i,:)); %10 x 1   1 x 26
Theta1_grad = Theta1_grad + d2 * (A1(i,:)); %25 x 1    
end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

%-----------------regularize-----------------

tmp2 = Theta2;
tmp2(:,1) = 0;

tmp1 = Theta1;
tmp1(:,1) = 0;

Theta2_grad = Theta2_grad + lambda/m * tmp2;
Theta1_grad = Theta1_grad + lambda/m * tmp1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
