function ksp = sake2(data,varargin)
% ksp = sake2(data,varargin)
%
% 2D MRI reconstruction based on matrix completion.
%
% Singular value filtering is done based on opts.std
% which is a key parameter affecting the image quality.
%
% Conjugate symmetry requires the center of kspace to be
% at the center of the array so that flip works correctly.
% Ditto for separate calibration data.
%
% Inputs:
%  -data [nx ny nc]: 2D kspace data array from nc coils
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nc]: 2D kspace data array from nc coils
%
% References:
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom
    data = fftshift(fft2(data));
    mask = false(256,256);
    R = 3; mask(:,1:R:end) = 1; % undersampling
    %mask(:,124:134) = 1; % self calibration (or separate)
    varargin{1} = 'width'; varargin{2} = R+2; % specify kernel width   
    varargin{3} = 'cal'; varargin{4} = data(:,124:134,:); % separate calibation
    %varargin{5} = 'loraks'; varargin{6} = 1; % employ conjugate symmetry 
    %data=1e-6*complex(randn(size(data)),randn(size(data)));
    data = bsxfun(@times,data,mask); clearvars -except data varargin
end

%% setup

% default options
opts.width = 5; % kernel width
opts.radial = 1; % use radial kernel
opts.loraks = 0; % conjugate coils (loraks)
opts.maxit = 1e4; % maximum no. iterations
opts.tol = 1e-5; % convergence tolerance
opts.p = 2; % singular filter shape (>=1) 
opts.std = []; % noise std dev, if available
opts.cal = []; % separate calibration data, if available
opts.sparsity = 0; % sparsity in wavelet domain (0.1=10% zeros)

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        if isempty(varargin{k}); continue; end
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        warning('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% initialize

% argument checks
if ndims(data)<2 || ndims(data)>3 || ~isfloat(data) || isreal(data)
    error('''data'' must be a 3d complex float array.')
end
if numel(opts.width)~=1
    error('width must be scalar');
end
[nx ny nc] = size(data);

% convolution kernel indicies
[x y] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = hypot(x,y)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.mask = k;

% estimate center of kspace
[~,k] = max(reshape(data,[],nc));
[x y] = ind2sub([nx ny],k);
opts.center(1) = gather(round(median(x)));
opts.center(2) = gather(round(median(y)));

% estimate noise std (heuristic)
if isempty(opts.std)
    tmp = nonzeros(data); tmp = sort([real(tmp); imag(tmp)]);
    k = ceil(numel(tmp)/10); tmp = tmp(k:end-k+1); % trim 20%
    opts.std = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2);
end
noise_floor = opts.std * sqrt(nnz(data)/nc);

% conjugate symmetric coils
if opts.loraks
    nc = 2*nc;
    data = cat(3,data,conj(flip(flip(data,1),2)));
    opts.cal = cat(3,opts.cal,conj(flip(flip(opts.cal,1),2)));
end

% dimensions of the dataset
opts.dims = [nx ny nc nk];

% set up wavelet transform
if opts.sparsity; Q = HWT([nx ny]); end

% display
disp(rmfield(opts,{'kernel'}));
fprintf('Density = %f\n',nnz(data)/numel(data));

%% see if gpu is possible

try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
    data = gpuArray(data);
    fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    warning('%s Using CPU.', ME.message);
end

%% separate calibration data

if ~isempty(opts.cal)

    if size(opts.cal,3)~=nc
        error('Calibration data has %i coils (data has %i).',size(opts.cal,3),nc);
    end

    A = make_data_matrix(cast(opts.cal,'like',data),opts);
    [V S] = svd(A'*A); S = sqrt(diag(S));
    
end

%% Cadzow algorithm

ksp = data;

for iter = 1:opts.maxit

    % make calibration matrix
    if iter==1
        A = make_data_matrix(ksp,opts);
        ix = (A ~= 0); val = A(ix);
    else
        A(ix) = val; % data consistency
    end
    
    % row space and singular values
    if isempty(opts.cal)
        [V S] = svd(A'*A);
        S = sqrt(diag(S));
    else
        S = sqrt(svd(A'*A));  
    end
    
    % singular value filter
    f = max(0,1-noise_floor.^opts.p./S.^opts.p);
    A = A * (V * diag(f) * V');
    
    % undo hankel structure
    if iter==1
        xi = 1:uint32(numel(A));
        if isa(ix,'gpuArray')
            xi = gpuArray(xi);
        end
        xi = undo_data_matrix(xi,opts);
    end
    ksp = mean(A(xi),4);

    % sparsity
    if opts.sparsity
        ksp = fft2(ksp); % to image
        ksp = Q.thresh(ksp,opts.sparsity);
        ksp = ifft2(ksp); % to kspace
    end

    % schatten p-norm
    snorm(iter) = norm(S,opts.p);
    if iter<10 || snorm(iter)<snorm(iter-1)
        tol = NaN;
    else
        tol = (snorm(iter)-snorm(iter-1)) / snorm(iter);
    end

    % display progress every 1 second
    if iter==1 || toc(t(1)) > 1 || tol<opts.tol || iter==opts.maxit
        if iter==1
            display(S,f,noise_floor,ksp,iter,snorm,tol,opts); t(1:2) = tic();
        elseif t(1)==t(2)
            fprintf('Iterations per second: %.2f\n',(iter-1) / toc(t(1)));
            display(S,f,noise_floor,ksp,iter,snorm,tol,opts); t(1) = tic();
        else
            display(S,f,noise_floor,ksp,iter,snorm,tol,opts); t(1) = tic();
        end
    end

    % finish when nothing left to do
    if tol<opts.tol; break; end

end

fprintf('Iterations performed: %i (%.1f sec)\n',iter,toc(t(2)));
if nargout==0; clear; end % avoid dumping to screen

%% make data matrix
function A = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nc = size(data,3);
nk = opts.dims(4);

A = zeros(nx,ny,nc,nk,'like',data);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k) = circshift(data,[x y]);
end

A = reshape(A,nx*ny,nc*nk);

%% undo data matrix
function A = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(4);

A = reshape(A,nx,ny,nc,nk);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k) = circshift(A(:,:,:,k),-[x y]);
end

%% show plots of various things
function display(S,f,noise_floor,ksp,iter,snorm,tol,opts)

% plot singular values
subplot(1,4,1); plot(S/S(1)); title(sprintf('rank %i/%i',nnz(f),numel(f)));
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]); grid on;
line(xlim,gather([1 1]*noise_floor/S(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% prefer ims over imagesc
if exist('ims','file'); imagesc = @(x)ims(x,-0.99); end

% show current kspace
subplot(1,4,2); imagesc(log(sum(abs(ksp),3)));
xlabel('dim 2'); ylabel('dim 1'); title('kspace');
line(xlim,[opts.center(1) opts.center(1)]);
line([opts.center(2) opts.center(2)],ylim);

% show current image
subplot(1,4,3); imagesc(sum(abs(ifft2(ksp)),3));
xlabel('dim 2'); ylabel('dim 1'); title(sprintf('iter %i',iter));

% plot change in metrics
subplot(1,4,4); plot(snorm); xlabel('iters'); xlim([0 iter+1]); grid on;
title(sprintf('tol %.2e',tol)); legend('||A||_p','location','northwest');

drawnow;
