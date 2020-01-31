clear;
data=load('nuclear.mat');

X = data.x;
Y = data.y;
[m,n] = size(X);
X = X';
Y = Y';

X = [ones(size(X,1),1) X];

num_iters = 100;
lambda = .001;
diff = 0.0001;
bs = 1;

obj = zeros(num_iters,1);
theta = zeros(m+1,1);

% meshgrid for visualization
xrange = linspace(min(X(:,2)), max(X(:,2)), 1000);
yrange =  linspace(min(X(:,3)), max(X(:,3)), 1000);
[xs,ys] =meshgrid(xrange, yrange);

for epoch=1:num_iters
    % visualize the boundary on data
    hold off;
    z = [ ones(numel(xs),1) xs(:) ys(:)]*theta;
    z = reshape(z, size(xs));
    imagesc(xrange, yrange, z<0);
    hold on
    plot(X(Y==1,2), X(Y==1,3), 'bo');
    plot(X(Y==-1,2), X(Y==-1,3), 'ro');
    pause(.1);
    
    r =randperm(n);
    iter = n/bs;

    for i = 1:iter
        index = r(:,i);
        x = X(index,:);
        y = Y(index,:);
        v = y*(x*theta);
        weight = [0; theta(2,1); theta(3,1)];
        if 1- v > 0
            theta = theta - (100/(epoch*n))*((-1)*y*x'+ lambda*weight);
        else
            theta = theta - (100/(epoch*n))*lambda*weight;
        end
    end
    
    h = max(0,1-Y.*(X*theta));
    weight_obj = [0; theta(2,1); theta(3,1)];
    obj(epoch) = (1/n)*sum(h)+ (lambda/2)*(norm(weight_obj))^2;
    
    if epoch > 1
       cond = abs(obj(epoch-1)-obj(epoch));
       if cond < diff
           break
       end
    end
    
end

disp(theta);
h = max(0,1-Y.*(X*theta));
weight_final = [0; theta(2,1); theta(3,1)];
Objective = (1/n)*sum(h)+ (lambda/2)*(norm(weight_final))^2;
disp(Objective);

figure, plot(obj);
xlabel('iterations');
ylabel('Objective');



