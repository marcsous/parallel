function out = fft3(in,m,n,p)

if nargin<2; m = size(in,1); end
if nargin<3; n = size(in,2); end
if nargin<4; p = size(in,3); end
out = fft2(in,m,n);
out = fft(out,p,3);
