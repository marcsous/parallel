function C = cconvn(A,B)

% cconvn  N-dimensional circular convolution

sA = size(A);
sB = size(B);

% indices of wrapped endpoints
for k = 1:numel(sA)
    if sA(k)==1 || k>numel(sB) || sB(k)==1
        s{k} = ':';
    else
        s{k} = [sA(k)-ceil(sB(k)/2)+2:sA(k) 1:sA(k) 1:floor(sB(k)/2)];
    end
end

% pad array
A = A(s{:});

% convn valid
C = convn(A,B,'valid');

