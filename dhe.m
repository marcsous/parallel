function [ksp basic] = dhe(fwd,rev,varargin)
% [ksp basic] = dhe(fwd,rev,varargin)
%
% Double Half Echo Reconstruction
%
% fwd = 2D kspace with forward readout direction
% rev = 2D kspace with reverse readout direction
%
% ksp is the reconstructed kspace for fwd/rev
% basic is a basic non-low rank reconstruction
%
%% example dataset

if nargin==0
    disp('Running example...')
    load partial_echo.mat
    varargin = {'sigma',5e-6,'loraks',1};
end

%% setup

% default options
opts.width = 5; % kernel width
opts.radial = 1; % use radial kernel
opts.loraks = 0; % conjugate symmetry
opts.tol = 1e-6; % tolerance (fraction change in norm)
opts.maxit = 1e4; % maximum no. iterations
opts.sigma = []; % noise std, if available
opts.removeOS = 0; % remove 2x oversampling in kx
opts.delete1stpoint = 1; % delete 1st readout point
opts.readout = 1; % readout dimension (1 or 2)

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        warning('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% initialize

% argument checks
if ndims(fwd)<2 || ndims(fwd)>3 || ~isfloat(fwd)
    error('''fwd'' must be a 3d float array.')
end
if ndims(rev)<2 || ndims(rev)>3 || ~isfloat(rev)
    error('''rev'' must be a 3d float array.')
end
if ~isequal(size(fwd),size(rev))
    error('''fwd'' and ''rev'' must be same size.')
end
if opts.readout==2
    fwd = permute(fwd,[2 1 3]);
    rev = permute(rev,[2 1 3]);  
elseif opts.readout~=1
    error('readout must be 1 or 2');
end
[nx ny nc] = size(fwd);

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

% dimensions of the dataset (4th dim for fwd/rev)
opts.dims = [nx ny nc 2 nk 1];
if opts.loraks; opts.dims(6) = 2; end

% create data and mask array
data = cat(4,fwd,rev);
mask = any(data,3);

% fractional echo length
fx = sum(any(any(mask,2),3));

% delete 1st readout point (ADC "warm up")
if opts.delete1stpoint
    samples = any(any(mask,2),3);
    overlap = find(sum(samples,4)==2);
    if numel(overlap)<=1
        % not enough pts to delete
    elseif samples(min(overlap)-1,1)
        mask(min(overlap),:,:,2) = 0;
        mask(max(overlap),:,:,1) = 0;
    else
        mask(min(overlap),:,:,1) = 0;
        mask(max(overlap),:,:,2) = 0;
    end
end

% estimate center of kspace
[~,k] = max(reshape(abs(data),[],nc,2));
[x y] = ind2sub([nx ny],reshape(k,nc,2));
center = round([median(x,1);median(y,1)]); % median over coils
opts.center = gather(round(mean(center,2)))'; % mean of fwd/rev

% align center of kx (necessary for loraks)
if opts.loraks
    for k = 1:2
        data(:,:,:,k) = circshift(data(:,:,:,k),opts.center(1)-center(1,k));
        mask(:,:,:,k) = circshift(mask(:,:,:,k),opts.center(1)-center(1,k)); 
    end
end

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);

% density of data matrix
matrix_density = nnz(mask) / numel(mask);

% display
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Density = %f\n',matrix_density);
if opts.loraks
    fprintf('Shifted [fwd/rev] by [%+i/%+i]\n',opts.center(1)-center(1,:));
end
fprintf('fwd: [%i/%i] rev: [%i/%i]\n',fx(1),nx,fx(2),nx);

%% see if gpu is possible

try
    gpu = gpuDevice;  
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
    fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);    
    fwd = gpuArray(fwd);
    rev = gpuArray(rev);
    mask = gpuArray(mask);
catch ME
    warning('%s Using CPU.', ME.message);    
    fwd = gather(fwd);
    rev = gather(rev);
    mask = gather(mask);
end

%% basic algorithm (average in place)

basic = sum(data,4)./max(sum(mask,4),1);

%% Cadzow algorithm

ksp = cat(4,basic,basic);

for iter = 1:opts.maxit

    % data consistency
    ksp = ksp + bsxfun(@times,data-ksp,mask);
    
    % data matrix
    A = make_data_matrix(ksp,opts);

    % row space and singular values (squared)
    [V W] = svd(A'*A);
    W = diag(W);

    % estimate noise std (heuristic)
    if isempty(opts.sigma)
        hi = nnz(W > eps(numel(W)*W(1))); % skip true zeros
        for lo = 1:hi
            h = hist(W(lo:hi),sqrt(hi-lo));
            [~,k] = max(h);
            if k>1; break; end
        end
        opts.sigma = sqrt(median(W(lo:hi))/matrix_density/nx/ny);
        fprintf('Noise std estimate: %.2e\n',opts.sigma);
    end
    noise_floor = opts.sigma * sqrt(matrix_density*nx*ny);

    % unsquare singular values
    W = sqrt(gather(W));   
    
    % minimum variance filter
    f = max(0,1-noise_floor^2./W.^2);
    A = A * (V * diag(f) * V');
    
    % hankel structure (average along anti-diagonals)   
    ksp = undo_data_matrix(A,opts);
    
    % keep track of norms
    norms(1,iter) = norm(W,1); % nuclear norm 
    norms(2,iter) = norm(W,2); % Frobenius norm
    
    % check convergence (fractional change in Frobenius norm)
    if iter==1
        tol(iter) = opts.tol;
    else
        tol(iter) = abs(norms(2,iter)-norms(2,iter-1))/norms(2,iter);
    end
    converged = sum(tol<opts.tol) > 10;

    % display progress every few iterations
    if mod(iter,10)==1 || converged
        display(W,f,noise_floor,ksp,iter,norms,tol,mask,opts,converged);
    end
    if iter==1
        t = tic();
    elseif iter==3
        fprintf('Time per iteration: %.1f ms\n',1000*toc(t)/(iter-1));
    end

    % finish when nothing left to do
    if converged; break; end

end

% remove oversampling
if opts.removeOS
    ok = nx/4 + (1:nx/2);
    ksp = fftshift(ifft(ksp,[],1));
    ksp = ksp(ok,:,:,:);
    ksp = fft(ifftshift(ksp),[],1);
    
    basic = fftshift(ifft(basic,[],1));
    basic = basic(ok,:,:,:);
    basic = fft(ifftshift(basic),[],1);
end

% restore original orientation
if opts.readout==2
    fwd = permute(fwd,[2 1 3]);
    rev = permute(rev,[2 1 3]);
    basic = permute(basic,[2 1 3]);
end

if nargout==0; clear; end % avoid dumping to screen

%% make data matrix
function A = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nc = size(data,3);
nk = opts.dims(5);

A = zeros(nx,ny,nc,2,nk,'like',data);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,:,k) = circshift(data,[x y]);
end

if opts.loraks
    A = cat(6,A,conj(A(opts.flip.x,opts.flip.y,:,:,:)));
end

A = reshape(A,nx*ny,[]);

%% undo data matrix
function data = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(5);

A = reshape(A,nx,ny,nc,2,nk,[]);

if opts.loraks
    A(opts.flip.x,opts.flip.y,:,:,:,2) = conj(A(:,:,:,:,:,2));
end

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,:,k,:) = circshift(A(:,:,:,:,k,:),-[x y]);
end

data = mean(reshape(A,nx,ny,nc,2,[]),5);

%% show plots of various things
function display(W,f,noise_floor,ksp,iter,norms,tol,mask,opts,converged)

% plot singular values
subplot(2,4,1); plot(W/W(1)); title(sprintf('rank %i/%i',nnz(f),numel(f))); 
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]);
line(xlim,gather([1 1]*noise_floor/W(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% plot change in metrics
subplot(2,4,5);
ax = plotyy(1:iter,norms(1,:),1:iter,1./norms(2,:));
legend('||A||_*','||A||_F^{-1}'); axis(ax,'tight');
xlabel('iters'); title(sprintf('tol %.2e',tol(end)));
drawnow;

% mask on iter=1 to show the blackness of kspace
if iter==1
    ksp = bsxfun(@times,ksp,mask);
end

% prefer ims over imagesc
if exist('ims','file'); imagesc = @(x)ims(x,-0.99); end

% show current kspace
subplot(2,4,2); imagesc(log(sum(abs(ksp(:,:,:,1)),3)));
xlabel(num2str(size(ksp,2),'ky [%i]'));
ylabel(num2str(size(ksp,1),'kx [%i]'));
title('kspace (fwd)');
subplot(2,4,6); imagesc(log(sum(abs(ksp(:,:,:,2)),3)));
xlabel(num2str(size(ksp,2),'ky [%i]'));
ylabel(num2str(size(ksp,1),'kx [%i]'));
title('kspace (rev)');

% remove oversampling
if opts.removeOS
    nx = size(ksp,1);
    ok = nx/4 + (1:nx/2);
    ksp = fftshift(ifft(ksp,[],1));
    ksp = ksp(ok,:,:,:);
    ksp = fft(ifftshift(ksp),[],1);
end

% show current image 
subplot(2,4,3); imagesc(fftshift(sum(abs(ifft2(ksp(:,:,:,1))),3)));
xlabel('y'); ylabel('x'); title(sprintf('iter %i (fwd)',iter));
subplot(2,4,7); imagesc(fftshift(sum(abs(ifft2(ksp(:,:,:,2))),3)));
xlabel('y'); ylabel('x'); title(sprintf('iter %i (rev)',iter));

% show one coil image phase
subplot(2,4,4); imagesc(fftshift(angle(ifft2(fftshift(ksp(:,:,1,1))))));
xlabel('y'); ylabel('x'); title(sprintf('phase (fwd)'));
subplot(2,4,8); imagesc(fftshift(angle(ifft2(fftshift(ksp(:,:,1,2))))));
xlabel('y'); ylabel('x'); title(sprintf('phase (rev)'));
drawnow;
