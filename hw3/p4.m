clear;
data=load('mnist_49_3000.mat');

x = data.x;
y = data.y;

[m,n] = size(x);
x = [ones(1,n);x];

train_size = 2000;

train_x = x(:,1:train_size);
train_y = y(:,1:train_size);
test_x = x(:,train_size+1:n);
test_y = y(:,train_size+1:n);

iter = 0;
max_iter = 10;

lambda = 10;
diff = 0.000001;
theta = zeros(m+1,1);
Objective_old = Inf;

while iter < max_iter
    iter = iter + 1;
    
    log_likelihood = 0;
    sum_g = zeros(m+1,1);
    sum_h = zeros(m+1,m+1);
    
    for i = 1:train_size
        log_likelihood = log_likelihood +( (-1)*log(1+exp((-1)*train_y(:,i) * theta' * train_x(:,i))));
        exp_p = exp(train_y(:,i) * theta' * train_x(:,i));
        sum_g = sum_g + ((-1)* train_x(:,i)* train_y(:,i)/(1+exp_p));
        sum_h = sum_h + train_x(:,i)*train_x(:,i)'*train_y(:,i)*train_y(:,i)*exp_p/((1+exp_p)^2);
    end

    g = sum_g+2*lambda*theta;
    h = sum_h+2*lambda*eye(m+1);
    theta = theta - (h)^(-1)*g;
    
    Objective = (-1)*log_likelihood+lambda*((norm(theta))^2);
    if abs(Objective_old - Objective) < diff
        break
    end
    
    Objective_old = Objective;
end

y_hat = theta'*test_x;

count = 0;
for i = 1:n-train_size
    if sign(y_hat(1,i))~=test_y(1,i)
        count = count+1;
    end
end

err = count/(n-train_size);
fprintf('Test Error: %.4f \n', err);
fprintf('The objective function at the optimum: %.4f \n', Objective_old);

misclass = zeros(count,2);
index = 1;
for i = 1:n-train_size
    if sign(y_hat(1,i))~=test_y(1,i)
        misclass(index,:) = [i,abs(y_hat(1,i))];
        index = index+1;
    end
end

misclass = sortrows(misclass,2,'descend');

for i = 1:20
    id = misclass(i,1);
    subplot(4,5,i);
    img = reshape(test_x(2:785,id),[28,28]);
    imshow(img);
    if test_y(1,id) == 1
        title('true label = 9');
    else
        title('true label = 4');
    end
end



