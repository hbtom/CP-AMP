function ainr = yqtensor(W)

I = numel(W);

if I == 1
    ainr = W{1};
end

if I == 2
    [M, ~] = size(W{1});
    [N, R] = size(W{2});
    ainr = zeros(M,N,R);
    for r=1:R
        ainr(:,:,r) =  W{1}(:,r)*W{2}(:,r)';
    end
end

if I >= 3
    ainr = [];
    [~, R] = size(W{1});
    
    for r=1:R
        X = cell(1,I);
        for j = 1:I
            X(j) = eval(['{W{',num2str(j),'}(:,r)}']);
        end
        ainr1 = double(ktensor(1,X));
        if r == 1
            ainr = ainr1;
        else
            ainr = cat(I+1, ainr, ainr1);
        end
    end
end



