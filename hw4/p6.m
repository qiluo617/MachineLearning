clear;
data=load('bodyfat_data.mat');

x = data.X;
y = data.y;

lambda = 0.003;
theta = 15;

[m,n] = size(x);
train_size = 150;
test_size = m - train_size;

train_x = x(1:train_size,:);
train_y = y(1:train_size,:);
test_x = x(train_size+1:m,:);
test_y = y(train_size+1:m,:);

y_bar = mean(train_y);
y_tail = train_y - y_bar;

O = (1/train_size) * ones(train_size,train_size);
K_gauss=zeros(train_size,train_size);
for j=1:train_size
    for i=1:train_size
        K_gauss(i,j)= exp(((-1)/(2*theta^2))*(norm(train_x(j,:)-train_x(i,:)))^2);
    end
end

K_tail = K_gauss - K_gauss*O - O*K_gauss + O*K_gauss*O; 

f_gauss = zeros(train_size,1);
for i=1:train_size
    p = K_tail(:,i);
    f_gauss(i,1) = y_bar + y_tail'*((K_tail+train_size*lambda*eye(train_size))\p);
end

mse_train =  norm(f_gauss-train_y)^2/train_size;

disp('With offset');
disp('MSE for training:');
disp(mse_train);

w=zeros(train_size,1);
for j=1:train_size
    u = train_x(j,:) - mean(train_x);
    v = mean(train_x);
    w(j,1)= exp(((-1)/(2*theta^2))*(norm(u-v))^2);
end
w_tx= y_tail'*((K_tail+train_size*lambda*eye(train_size))\w);
b = y_bar -w_tx;

disp('b value:');
disp(b);

k_prime = zeros(train_size,test_size);
for j=1:train_size
    for i=1:test_size
        k_prime(j,i)= exp(((-1)/(2*theta^2))*(norm(train_x(j,:)-test_x(i,:)))^2);
    end
end

N = O;
M = (1/test_size)*ones(test_size, test_size);

k_prime_tail = k_prime - N*k_prime - k_prime*M + N*k_prime*M;

f_gauss_test = zeros(test_size,1);
for i=1:test_size
    p_test = k_prime_tail(:,i);
    inner =(K_tail+train_size*lambda*eye(train_size))\p_test;
    f_gauss_test(i,1) = y_bar + y_tail'*(inner);
end

mse_test =  norm(f_gauss_test-test_y)^2/test_size;

disp('MSE for testing:');
disp(mse_test);

disp('Without offset');
f_train = zeros(train_size,1);
for i=1:train_size
    k_x = K_gauss(:,i);
    f_train(i,1) = train_y'*((K_gauss + train_size*lambda*eye(train_size))\k_x);
end

mse_train_nooffset =  norm(f_train-train_y)^2/train_size;

disp('MSE for training:');
disp(mse_train_nooffset);

k = zeros(train_size,test_size);
for j=1:train_size
    for i=1:test_size
        k(j,i)= exp(((-1)/(2*theta^2))*(norm(train_x(j,:)-test_x(i,:)))^2);
    end
end

f_test = zeros(test_size,1);

for i=1:test_size
    k_x_test = k(:,i);
    f_test(i,1) = train_y'*((K_gauss + train_size*lambda*eye(train_size))\k_x_test);
end

mse_test_nooffset =  norm(f_test-test_y)^2/test_size;

disp('MSE for testing:');
disp(mse_test_nooffset);

