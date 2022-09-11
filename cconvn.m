function C = cconvn2(A,B)

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

% pad array for 'valid'
switch numel(sA)
    case 1; A = A(s{1});
    case 2; A = A(s{1},s{2});
    case 3; A = A(s{1},s{2},s{3});
    case 4; A = A(s{1},s{2},s{3},s{4});
    case 5; A = A(s{1},s{2},s{3},s{4},s{5});
    case 6; A = A(s{1},s{2},s{3},s{4},s{5},s{6});
    case 7; A = A(s{1},s{2},s{3},s{4},s{5},s{6},s{7});
    case 8; A = A(s{1},s{2},s{3},s{4},s{5},s{6},s{7},s{8});       
    otherwise; error('high dimension not supported - fix me');
end

% convn valid
C = convn(A,B,'valid');

