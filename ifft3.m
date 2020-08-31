function arg = ifft3(arg,m,n,p,varargin)

if nargin<2; m = size(arg,1); end
if nargin<3; n = size(arg,2); end
if nargin<4; p = size(arg,3); end

arg = ifft(arg,m,1,varargin{:});
arg = ifft(arg,n,2,varargin{:});
arg = ifft(arg,p,3,varargin{:});
