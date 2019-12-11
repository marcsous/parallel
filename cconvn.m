function B = cconvn(A,K)
%
% Convolution of N-D array with kernel K.
% Performs circular wrapping at edges.
%
% A is an N-D array
% K is a kernel

% argument checks
if nargin<1 || isempty(A)
    error('something wrong with A');
end
if nargin<2 || isempty(K)
    error('something wrong with K');
end

% kernel properties
sz = size(K);
[i,~,v] = find(K(:)); % non-zeros
nd = ndims(K)-iscolumn(K); % no. dimensions

% bail out of unsupported cases
for d = 1:nd
    if sz(d)>size(A,d)
        error('dimension %i of A is smaller than K - fix me',d);
    end
end

% shift indicies
switch nd
    case 1; [S(1,:)] = ind2sub(sz,i);
    case 2; [S(1,:) S(2,:)] = ind2sub(sz,i);
    case 3; [S(1,:) S(2,:) S(3,:)] = ind2sub(sz,i);
    case 4; [S(1,:) S(2,:) S(3,:) S(4,:)] = ind2sub(sz,i);
    case 5; [S(1,:) S(2,:) S(3,:) S(4,:) S(5,:)] = ind2sub(sz,i);      
    otherwise; error('high dimensions not implemented - fix me');
end

% center the convolution indices
for d = 1:nd
    S(d,:) = S(d,:)-fix(sz(d)/2)-1; 
end

% perform the convolution
B = zeros(size(A),'like',A);

for k = 1:numel(v)
    B = B + circshift(A,S(:,k)) * v(k);
end
