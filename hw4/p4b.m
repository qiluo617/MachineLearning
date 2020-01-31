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
obj = zeros(num_iters,1);

% meshgrid for visualization
xrange = linspace(min(X(:,2)), max(X(:,2)), 1000);
yrange =  linspace(min(X(:,3)), max(X(:,3)), 1000);
[xs,ys] =meshgrid(xrange, yrange);

theta = zeros(m+1,1);
for iter=1:num_iters
    % visualize the boundary on data
    hold off;
    z = [ ones(numel(xs),1) xs(:) ys(:)]*theta;
    z = reshape(z, size(xs));
    imagesc(xrange, yrange, z<0);
    hold on
    plot(X(Y==1,2), X(Y==1,3), 'bo');
    plot(X(Y==-1,2), X(Y==-1,3), 'ro');
    pause(.1);

    grad = zeros(1,m+1);
    for i = 1:n
        v = Y(i,:)*(X(i,:)*theta);
        weight = [0; theta(2,1); theta(3,1)];
        if 1- v > 0
            sub_grad = (1/n)*((-1)*Y(i,:)*X(i,:) + lambda*weight');
            grad = grad + sub_grad;
        else
            grad = grad + (1/n)*lambda*weight';
        end
    end

    theta = theta - (100/iter)*grad';
    
    h = max(0,1-Y.*(X*theta));
    weight_obj = [0; theta(2,1); theta(3,1)];
    obj(iter) = (1/n)*sum(h)+ (lambda/2)*(norm(weight_obj))^2;
    
    if iter > 1
       cond = abs(obj(iter-1)-obj(iter));
       if cond < diff
           break
       end
    end
end
disp(theta);
h = max(0,1-Y.*(X*theta));
final_weight = [0; theta(2,1); theta(3,1)];
Objective = (1/n)*sum(h)+ (lambda/2)*(norm(final_weight))^2;
disp(Objective);
figure, plot(obj);
xlabel('iterations');
ylabel('Objective');