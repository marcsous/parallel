function arg = fft3(arg,m,n,p)

if nargin<2; m = size(arg,1); end
if nargin<3; n = size(arg,2); end
if nargin<4; p = size(arg,3); end

arg = fft(arg,m,1);
arg = fft(arg,n,2);
arg = fft(arg,p,3);
