function ksp = sake2(data,varargin)
% ksp = sake2(data,varargin)
%
% 2D MRI reconstruction based on matrix completion.
%
% Singular value filtering is done based on opts.noise
% which is a key parameter that affects image quality.
%
% Conjugate symmetry requires the center of kspace to be
% at the center of the array so that flip works correctly.
% Same for separate calibration (but doesn't work well).
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
    load phantom.mat


%     load phantom_32ch_7ech_1sl.mat
%     data=data(:,:,1:3:end,:);
%     %data=circshift(data,[2 0]);
%     cal=data(:,85:108,:,1); % beyond echo 3 not good even for sake alone
%     data=data(:,:,:,7);
% data(109:end,:,:)=0;
% 
% varargin = {'cal',cal};

    data = fftshift(fft2(data));
    mask = false(256,256);
    mask(1:141,1:2:256) = 1; % undersampling (and partial kx)
    mask(1:141,121:136) = 1; % self-calibration 
    %varargin{1} = 'cal'; varargin{2} = data(:,121:136,:); % separate calibation
    varargin{3} = 'loraks'; varargin{4} = 1; % assume conjugate symmetry
    data = bsxfun(@times,data,mask); clearvars -except data varargin
end

%% setup

% default options
opts.width = 4; % kernel width
opts.radial = 1; % use radial kernel
opts.loraks = 0; % conjugate coils (loraks)
opts.tol = 1e-7; % tolerance (fraction change in norm)
opts.maxit = 1e4; % maximum no. iterations
opts.noise = []; % noise std, if available
opts.cal = []; % separate calibration data, if available
opts.sparsity = 1; % sparsity in wavelet domain (1=100%)

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
if ndims(data)<2 || ndims(data)>3 || ~isfloat(data)
    error('''data'' must be a 3d float array.')
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
if isempty(opts.noise)
    tmp = nonzeros(data); tmp = sort([real(tmp); imag(tmp)]);
    k = ceil(numel(tmp)/10); tmp = tmp(k:end-k); % trim 20%
    opts.noise = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2);
end
noise_floor = opts.noise * sqrt(nnz(data)/nc);

% conjugate symmetric coils
if opts.loraks
    nc = 2*nc;
    data = cat(3,data,conj(flip(flip(data,1),2)));
    if ~isempty(opts.cal)
        opts.cal = cat(3,opts.cal,conj(flip(flip(opts.cal,1),2)));
    end
end

% dimensions of the dataset
opts.dims = [nx ny nc nk];

% set up DWT transform
if opts.sparsity<1 && opts.sparsity>0
    Q = DWT([nx ny],'db2'); % [nx ny] or [nx ny nc]?
elseif opts.sparsity~=1
    error('''sparsity'' must be between 0 and 1');
end

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

    cal = cast(opts.cal,'like',data);
    A = make_data_matrix(cal,opts);
    [V,~] = svd(A'*A);

end

%% Cadzow algorithm

mask = data ~= 0; % sampling mask
ksp = zeros(size(data),'like',data);

for iter = 1:opts.maxit

    % data consistency
    ksp = ksp + (data-ksp).*mask;

    % sparsity
    if opts.sparsity<1
        ksp = fft2(ksp); % to image
        ksp = Q.thresh(ksp,opts.sparsity);
        ksp = ifft2(ksp); % to kspace
    end

    % calibration matrix
    A = make_data_matrix(ksp,opts);

    % row space and singular values
    if isempty(opts.cal)
        [V W] = svd(A'*A);
        W = diag(W);
    else
        W = svd(A'*A);
    end
    W = sqrt(gather(W));

    % minimum variance filter
    f = max(0,1-noise_floor.^2./W.^2);
    A = A * (V * diag(f) * V');

    % hankel structure (average along anti-diagonals)
    ksp = undo_data_matrix(A,opts);

    % check convergence (fractional change in Frobenius norm)
    norms(1,iter) = norm(W,1); % nuclear norm
    norms(2,iter) = norm(W,2); % Frobenius norm
    if iter==1
        tol(iter) = opts.tol;
    else
        tol(iter) = abs(norms(2,iter)-norms(2,iter-1))/norms(2,iter);
    end
    converged = sum(tol<opts.tol) > 10;

    % display progress every 1 second
    if iter==1 || toc(t) > 1 || converged
         if exist('t','var') && ~exist('itspersec','var')
            itspersec = (iter-1)/toc(t);
            fprintf('Iterations per second: %.1f\n',itspersec);
        end
        display(W,f,noise_floor,ksp,iter,tol,norms,mask,opts);
        t = tic();
    end

    % finish when nothing left to do
    if converged; break; end

end

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
function data = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(4);

A = reshape(A,nx,ny,nc,nk);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k,:) = circshift(A(:,:,:,k,:),-[x y]);
end

data = mean(A,4);

%% show plots of various things
function display(W,f,noise_floor,ksp,iter,tol,norms,mask,opts)

% plot singular values
subplot(1,4,1); plot(W/W(1)); title(sprintf('rank %i/%i',nnz(f),numel(f)));
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]);
line(xlim,gather([1 1]*noise_floor/W(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% mask on iter=1 to show the blackness of kspace
if iter==1
    ksp = ksp .* mask; 
    if opts.loraks; ksp = ksp(:,:,1:size(ksp,3)/2); end
end

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
subplot(1,4,4);
ax = plotyy(1:iter,norms(1,:),1:iter,norms(2,:));
legend('||A||_*','||A||_F');axis(ax,'tight');
xlabel('iters'); title(sprintf('tol %.2e',tol(end)));
drawnow;
