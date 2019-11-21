function out = fft3(in,m,n,p)

if nargin<2; m = size(in,1); end
if nargin<3; n = size(in,2); end
if nargin<4; p = size(in,3); end
out = fft(in ,m,1);
out = fft(out,n,2);
out = fft(out,p,3);
