function ksp = sake3(data,varargin)
% ksp = sake3(data,varargin)
%
% 3D MRI reconstruction based on matrix completion.
% Low memory version does not form matrix but is slow.
%
% Singular value filtering is done based on opts.std
% which is a key parameter that affects image quality.
%
% Conjugate symmetry requires the center of kspace to be
% at the center of the array so that flip works correctly.
%
% Inputs:
%  -data [nx ny nz nc]: 3D kspace data array from nc coils
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nz nc]: 3D kspace data array from nc coils
%
% References:
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%
%% example dataset

if nargin==0 || isempty(data)
    disp('Running example...')
    % note: this isn't perfect... R=4 with 6coil is pushing it (dataset < 25Mb for github)
    load phantom3D_6coil.mat
    mask = false(size(data,1),size(data,2),size(data,3));
    mask(:,1:2:end,1:2:end) = 1; % undersample 2x2
    mask(:,3:4:end,:) = circshift(mask(:,3:4:end,:),[0 0 1]); % shifted
    %mask(size(data,1)/2+(-9:9),size(data,2)/2+(-9:9),:) = 1; % self calibration
    varargin{1} = 'cal'; varargin{2} = data(size(data,1)/2+(-9:9),size(data,2)/2+(-9:9),size(data,3)/2+(-9:9),:); % separate calibration
    data = bsxfun(@times,data,mask); clearvars -except data varargin
end

%% setup

% default options
opts.width = 3; % kernel width: scalar or [x y z]
opts.radial = 1; % use radial kernel [1 or 0]
opts.loraks = 0; % phase constraint (loraks)
opts.maxit = 1e3; % maximum no. iterations
opts.tol = 1e-4; % convergence tolerance
opts.pnorm = 2; % singular filter shape (>=1) 
opts.std = []; % noise std dev, if available
opts.cal = []; % separate calibration data, if available
opts.gpu = 1; % use GPU, if available (often faster without)
opts.sparsity = 0; % sparsity in wavelet domain (0.1=10% zeros)

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        if isempty(varargin{k}); continue; end
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        error('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% initialize

% argument checks
if ndims(data)<3 || ndims(data)>4 || ~isfloat(data) || isreal(data)
    error('''data'' must be a 4d complex float array.')
end
if numel(opts.width)==1
    opts.width = [1 1 1] * opts.width;
elseif numel(opts.width)~=3
    error('width must have 1 or 3 elements.');
end
[nx ny nz nc] = size(data);
if nz==1; opts.width(3) = 1; end % handle 2D case

% convolution kernel indicies
[x y z] = ndgrid(-ceil(opts.width(1)/2):ceil(opts.width(1)/2), ...
                 -ceil(opts.width(2)/2):ceil(opts.width(2)/2), ...
                 -ceil(opts.width(3)/2):ceil(opts.width(3)/2));
if opts.radial
    k = x.^2/max(1,opts.width(1)^2)+y.^2/max(1,opts.width(2)^2)+z.^2/max(1,opts.width(3)^2) <= 0.25;
else
    k = abs(x)/max(1,opts.width(1))<=0.5 & abs(y)/max(1,opts.width(2))<=0.5 & abs(z)/max(1,opts.width(3))<=0.5;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.z = z(k);
opts.kernel.mask = k;

% estimate center of kspace (heuristic)
[~,k] = max(reshape(data,[],nc));
[x y z] = ind2sub([nx ny nz],k);
opts.center(1) = gather(round(median(x)));
opts.center(2) = gather(round(median(y)));
opts.center(3) = gather(round(median(z)));

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
    data = cat(4,data,conj(flip(flip(flip(data,1),2),3)));
    if ~isempty(opts.cal)
        error('Conjugate symmetry not compatible with separate calibration.');
    end
end

% dimensions of the data set
opts.dims = [nx ny nz nc nk];

% set up wavelet transform:
if opts.sparsity; Q = HWT([nx ny nz]); end

% display
disp(rmfield(opts,{'kernel'}));
fprintf('Density = %f\n',nnz(data)/numel(data));

if ~nnz(data) || ~isfinite(noise_floor)
    error('data all zero or contains Inf/NaN.');
end

%% see if gpu is possible

try
    if ~opts.gpu; error('GPU option set to off.'); end
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
    gpu = gpuDevice; data = gpuArray(data);
    fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    warning('%s Using CPU.', ME.message);
end

%% separate calibration data

if ~isempty(opts.cal)
    
    if size(opts.cal,4)~=nc
        error('Calibration data has %i coils (data has %i).',size(opts.cal,4),nc);
    end

    cal = cast(opts.cal,'like',data);
    AA = make_data_matrix(cal,opts);
    [V S] = svd(AA); % could truncate V based on S...

end

%% Cadzow algorithm

mask = (data ~= 0); % sampling mask
ksp = zeros(size(data),'like',data);

for iter = 1:opts.maxit

    % data consistency
    ksp = ksp + bsxfun(@times,data-ksp,mask);
    
    % impose sparsity
    if opts.sparsity
        ksp = fft3(ksp); % to image
        ksp = Q.thresh(ksp,opts.sparsity);
        ksp = ifft3(ksp); % to kspace
    end
    
    % normal calibration matrix
    AA = make_data_matrix(ksp,opts);
    
    % row space and singular values
    if isempty(opts.cal)
        [V S] = svd(AA);
        S = sqrt(diag(S));
    else
        S = sqrt(svd(AA));
    end
    
    % singular value filter
    f = max(0,1-(noise_floor./S).^opts.pnorm); 
    F = V * diag(f) * V';
    
    % hankel structure (average along anti-diagonals)  
    ksp = undo_data_matrix(F,ksp,opts);
    
    % schatten p-norm of A
    snorm(iter) = norm(S,opts.pnorm); 
    if iter<10 || snorm(iter)<snorm(iter-1)
        tol = NaN;
    else
        tol = (snorm(iter)-snorm(iter-1)) / snorm(iter);
    end
    
    % display progress every 5 seconds
    if iter==1
        t(1:2) = tic(); % global/display timers
    elseif toc(t(2)) > 5 || tol < opts.tol
        if t(1)==t(2)
            fprintf('Iterations per second: %.2f\n',(iter-1)/toc(t(2)));
        end
        display(S,f,noise_floor,ksp,iter,tol,snorm,mask,opts); t(2) = tic();          
    end

    % finish when nothing left to do
    if tol < opts.tol; break; end
 
end

fprintf('Iterations performed: %i (%.0f sec)\n',iter,toc(t(1)));
if opts.gpu; ksp = gather(ksp); end % return on CPU
if nargout==0; clear; end % avoid dumping to screen

%% make normal calibration matrix (low memory)
function AA = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nz = size(data,3);
nc = size(data,4);
nk = opts.dims(5);

AA = zeros(nc,nk,nc,nk,'like',data);
  
for j = 1:nk

    x = opts.kernel.x(j);
    y = opts.kernel.y(j);
    z = opts.kernel.z(j);
    row = circshift(data,[x y z]); % rows of A.'
 
    for k = j:nk

        if j==k
            AA(:,j,:,k) = reshape(row,[],nc)' * reshape(row,[],nc);
        else
            x = opts.kernel.x(k);
            y = opts.kernel.y(k);
            z = opts.kernel.z(k);
            col = circshift(data,[x y z]); % cols of A

            % fill normal matrix (conjugate symmetric)
            tmp = reshape(row,[],nc)' * reshape(col,[],nc);
            AA(:,j,:,k) = tmp;
            AA(:,k,:,j) = tmp';
        end

    end

end
AA = reshape(AA,nc*nk,nc*nk);

%% undo calibration matrix (low memory)
function ksp = undo_data_matrix(F,data,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nz = opts.dims(3);
nc = opts.dims(4);
nk = opts.dims(5);

F = reshape(F,nc,nk,nc,nk);

ksp = zeros(nx,ny,nz,nc,'like',data);

for j = 1:nk

    x = opts.kernel.x(j);
    y = opts.kernel.y(j);
    z = opts.kernel.z(j);
    col = circshift(data,[x y z]); % cols of A
    
    for k = 1:nk

        tmp = reshape(col,[],nc) * reshape(F(:,j,:,k),nc,nc);
        tmp = reshape(tmp,nx,ny,nz,nc);

        % average along rows      
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        z = opts.kernel.z(k);
        ksp = ksp + circshift(tmp,-[x y z]) / nk;
    
    end
    
end

%% show plots of various things
function display(S,f,noise_floor,ksp,iter,tol,snorm,mask,opts)

% plot singular values
subplot(1,4,1); plot(S/S(1)); title(sprintf('rank %i',nnz(f))); 
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]); grid on;
line(xlim,gather([1 1]*noise_floor/S(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% mask on iter=1 to show the blackness of kspace
if iter==1
    ksp = ksp .* mask; 
    if opts.loraks; ksp = ksp(:,:,:,1:size(ksp,4)/2); end
end

% prefer ims over imagesc
if exist('ims','file'); imagesc = @(x)ims(x,-0.99); end

% show current kspace (center of kx)
subplot(1,4,2);
tmp = squeeze(log(sum(abs(ksp(opts.center(1),:,:,:)),4)));
imagesc(tmp); xlabel('kz'); ylabel('ky'); title('kspace');
line(xlim,[opts.center(2) opts.center(2)]);
line([opts.center(3) opts.center(3)],ylim);

% show current image (center of x)
subplot(1,4,3); slice = floor(size(ksp,1)/2+1); % middle slice in x
tmp = ifft(ksp); tmp = squeeze(tmp(slice,:,:,:));
imagesc(sum(abs(ifft2(tmp)),3)); xlabel('z'); ylabel('y');
title(sprintf('iter %i',iter));

% plot change in metrics
subplot(1,4,4); plot(snorm); xlabel('iters'); xlim([0 iter+1]); grid on;
title(sprintf('tol %.2e',tol)); legend('||A||_p','location','northwest');

drawnow;
