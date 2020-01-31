clear;
data = load('yalefaces.mat');
yalefaces = data.yalefaces;
shape = [size(yalefaces,1)*size(yalefaces,2),size(yalefaces,3)];
data = double(reshape(yalefaces, shape)');

mu = mean(data);
S = ((data - ones(size(data))* diag(mu))' * (data - ones(size(data))* diag(mu))) ./ size(yalefaces,3);
[U, D] = eig(S);
eig_values = sum(D);
[eig, index] = sort(eig_values,'descend');

figure(1);
semilogy(eig);

for i = 1:length(eig)
    var = sum(eig(:, 1:i)) / sum(eig);
    if var >= .95
        k95 = i;
        break
    end
end

for i = 1:length(eig)
    var = sum(eig(:, 1:i)) / sum(eig);
    if var >= .99
        k99 = i;
        break
    end
end

figure(2);
colormap(gray);
subplot(5,4, 1)
a = reshape(mu, size(yalefaces,1), size(yalefaces,2));
imagesc(a); 

for i = 2:20
    subplot(5, 4, i);
    x = reshape(U(:, index(i-1)), size(yalefaces,1), size(yalefaces,2));
    imagesc(x);
end