function [ksp basic nrm opts] = dhe(fwd,rev,varargin)
% [ksp basic nrm opts] = dhe(fwd,rev,varargin)
%
% Double Half Echo Reconstruction (2D only)
%
% fwd = kspace with forward readouts [nx ny nc ne]
% rev = kspace with reverse readouts [nx ny nc ne]
%
% In the interests of using the same recon to do
% comparisons, code accepts a single fwd dataset.
%
% -ksp is the reconstructed kspace for fwd/rev
% -basic is a basic non-low rank reconstruction
% -nrm is the convergence history (nuc and fro)
% -opts returns the options (opts.freq)
%
% Note: don't remove readout oversampling before
% calling this function. With partial sampling the
% fft+crop method is incorrect (specify opts.osf).
%
% Ref: https://dx.doi.org/10.1002/nbm.4458
%      https://doi.org/10.1016/j.mri.2022.08.017
%% example dataset

if nargin==0
    disp('Running example...')
    load meas_MID00382_FID42164_clean_fl2_m400.mat
    data = squeeze(data); fwd = data(:,:,1); rev = data(:,:,2);
    varargin = {'center',[],'delete1st',[2 0],'readout',2};
    clearvars -except fwd rev varargin
end

%% setup

% default options
opts.width = [5 5]; % kernel width (in kx ky)
opts.radial = 1; % use radial kernel
opts.loraks = 0; % conjugate symmetry
opts.tol = 1e-6; % relative tolerance
opts.gpu = 1; % use gpu if available
opts.maxit = 1e3; % maximum no. iterations
opts.std = []; % noise std dev, if available
opts.center = []; % center of kspace, if available
opts.delete1st = [1 0]; % delete [first last] readout pts
opts.readout = 1; % readout dimension (1 or 2)
opts.osf = 2; % readout oversampling factor (default 2)
opts.freq = []; % off resonance in deg/dwell ([] = auto)

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        error('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% argument checks
if ndims(fwd)<2 || ndims(fwd)>4 || ~isfloat(fwd) || isreal(fwd)
    error('''fwd'' must be a 2d-4d complex float array.')
end
if ~exist('rev','var') || isempty(rev)
    nh = 1; % no. half echos
    rev = []; % no rev echo
else
    nh = 2; % no. half echos
    if ndims(rev)<2 || ndims(rev)>4 || ~isfloat(rev) || isreal(fwd)
        error('''rev'' must be a 2d-4d complex float array.')
    end
    if ~isequal(size(fwd),size(rev))
        error('''fwd'' and ''rev'' must be same size.')
    end
end
if opts.osf<1
    error('osf must be >=1');
end
if mod(size(fwd,opts.readout)/2/opts.osf,1)
    error('readout dim (%i) not divisible by 2*osf.',size(fwd,opts.readout));
end
if opts.width(1)>size(fwd,1) || opts.width(end)>size(fwd,2)
    error('width [%ix%i] not compatible with matrix.\n',opts.width(1),opts.width(end));
end
if isscalar(opts.width)
    opts.width = [opts.width opts.width];
elseif opts.readout==2
    opts.width = flip(opts.width);
end
if opts.readout==2
    fwd = permute(fwd,[2 1 3 4]);
    rev = permute(rev,[2 1 3 4]);
elseif opts.readout~=1
    error('readout must be 1 or 2.');
end
if isequal(opts.std,0) || numel(opts.std)>1
    error('noise std must be a non-zero scalar');
end
if any(mod(opts.delete1st,1)) || any(opts.delete1st<0)
    error('delete1st must be a nonnegative integer.');
end
if isscalar(opts.delete1st)
    opts.delete1st = [opts.delete1st 0];
end
if ~isempty(opts.freq) && ~isscalar(opts.freq)
    error('freq must be scalar)');
end

%% initialize
[nx ny nc ne] = size(fwd);

% convolution kernel indicies
[x y] = ndgrid(-ceil(opts.width(1)/2):ceil(opts.width(1)/2), ...
               -ceil(opts.width(2)/2):ceil(opts.width(2)/2));
if opts.radial
    k = hypot(abs(x)/max(1,opts.width(1)),abs(y)/max(1,opts.width(2)))<=0.5;
else
    k = abs(x)/max(1,opts.width(1))<=0.5 & abs(y)/max(1,opts.width(2))<=0.5;
end
opts.kernel.x = x(k);
opts.kernel.y = y(k);
nk = nnz(k);

% dimensions of the dataset
opts.dims = [nx ny nc ne nh nk 1];
if opts.loraks; opts.dims(7) = 2; end

% concatenate fwd/rev echos
data = cat(5,fwd,rev);
mask = any(data,3);

% delete 1st (and last) ADC "warm up" points on kx
if any(opts.delete1st)
    for e = 1:ne
        for h = 1:nh
            kx = 1; init = any(mask(kx,:,1,e,h)); % is kx(1) sampled?
            while kx<nx && any(mask(kx,:,1,e,h))==init; kx = kx+1; end
            if init==1 % fwd echo
                mask(1:opts.delete1st(2),:,:,e,h) = 0;
                mask(kx:-1:max(kx-opts.delete1st(1),nx/2),:,:,e,h) = 0;
            else % rev echo
                mask(end:-1:end-opts.delete1st(2)+1,:,:,e,h) = 0;
                mask(kx:min(kx+opts.delete1st(1)-1,nx/2+1),:,:,e,h) = 0;
            end
        end
    end
    data = mask.*data;
end

% estimate center of kspace (heuristic)
if isempty(opts.center)
    [~,k] = max(reshape(abs(data),[],nc*ne,nh));
    [x y] = ind2sub([nx ny],reshape(k,nc*ne,nh));
    center = round([median(x,1);median(y,1)]); % for fwd and rev
    opts.center = gather(round(mean(center,2)))'; % mean of fwd/rev
elseif opts.readout==2
    opts.center = flip(opts.center);
end

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);

% estimate noise std (heuristic)
if isempty(opts.std)
    tmp = nonzeros(data); tmp = sort([real(tmp); imag(tmp)]);
    k = ceil(numel(tmp)/10); tmp = tmp(k:end-k+1); % trim 20%
    opts.std = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2);
end
noise_floor = opts.std * sqrt(nnz(mask));

% display
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Density = %f\n',nnz(mask)/numel(mask));
frac = sum(any(mask,2))/nx; % echo fraction
for j = 1:ne
    for k = 1:nh
        if k==1; txt = 'fwd'; else; txt = 'rev'; end
        fprintf('Echo fraction %i(%s): %.3f(%i)\n',j,txt,frac(1,1,1,j,k),round(frac(1,1,1,j,k)*nx));
    end
end

%% see if gpu is possible
if opts.gpu
    try
        gpu = gpuDevice; gpuArray(1); % trigger error if GPU is not working
        if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
        fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
        data = gpuArray(data);
        mask = gpuArray(mask);
        opts.flip.x = gpuArray(opts.flip.x);
        opts.flip.y = gpuArray(opts.flip.y);
    catch ME
        warning('%s Using CPU.', ME.message);
        data = gather(data);
        mask = gather(mask);
        opts.flip.x = gather(opts.flip.x);
        opts.flip.y = gather(opts.flip.y);
    end
end

%% corrections - need both fwd & rev
if nh>1 && ~isequal(opts.freq,0)

    % frequency: unit = deg/dwell
    opts.kx = (-nx/2:nx/2-1)' * pi / 180;

    % quick scan to find global minimum
    opts.range = linspace(-3,3,11);
    for k = 1:numel(opts.range)
        opts.nrm(k) = myfun(opts.range(k),data,opts);
    end
    [~,k] = min(opts.nrm); best = opts.range(k);

    % precalculate derivative matrix
    roll = cast(i*opts.kx,'like',data);
    tmp =           repmat(-roll,1,ny,nc,ne);
    tmp = cat(5,tmp,repmat(+roll,1,ny,nc,ne));
    tmp = reshape(tmp,size(data)); % make sure
    opts.P = make_data_matrix(tmp,opts);

    % off resonance (nuclear norm)
    if isempty(opts.freq)
        fopts = optimset('Display','off','GradObj','on');
        nrm = median(abs(nonzeros(data))); % mitigate poor scaling
        opts.freq = fminunc(@(f)myfun(f,data/nrm,opts),best,fopts);
    end

    % off resonance correction
    roll = exp(i*opts.kx*opts.freq);
    data(:,:,:,:,1) = data(:,:,:,:,1)./roll;
    data(:,:,:,:,2) = data(:,:,:,:,2).*roll;

    % phase correction
    r = dot(data(:,:,:,:,1),data(:,:,:,:,2));
    d = dot(data(:,:,:,:,1),data(:,:,:,:,1));
    r = reshape(r,[],1); d = reshape(real(d),[],1);
    phi = angle((r'*d) / (d'*d)) / 2;

    data(:,:,:,:,1) = data(:,:,:,:,1)./exp(i*phi);
    data(:,:,:,:,2) = data(:,:,:,:,2).*exp(i*phi);

    % units: phi=radians freq=deg/dwell
    fprintf('Corrections: ϕ=%.2frad Δf=%.2fdeg/dwell\n',phi,opts.freq);

    % clear memory on GPU
    opts.P = []; clear tmp roll r d nrm

end

%% basic algorithm (average in place)

basic = sum(data.*mask,5)./max(sum(mask,5),1);

%% Cadzow algorithm

ksp = zeros(size(data),'like',data);

for iter = 1:max(1,opts.maxit)

    % data consistency
    ksp = ksp + bsxfun(@times,data-ksp,mask);

    % make calibration matrix
    [A opts] = make_data_matrix(ksp,opts);

    % row space and singular values
    if size(A,1)<=size(A,2)
        [~,W,V] = svd(A,'econ');
        W = diag(W);
        V = V(:,1:numel(W));
    else
        [V W] = svd(A'*A);
        W = sqrt(diag(W));
    end

    % minimum variance filter
    f = max(0,1-noise_floor^2./W.^2);
    A = A * (V * diag(f) * V');

    % undo hankel structure
    [ksp opts] = undo_data_matrix(A,opts);

    % check convergence (fractional change in Frobenius norm)
    nrm(1,iter) = norm(W,1); % nuclear norm
    nrm(2,iter) = norm(W,2); % Frobenius norm
    if iter==1
        tol(iter) = cast(opts.tol,'like',nrm);
    else
        tol(iter) = abs(nrm(2,iter)-nrm(2,iter-1))/nrm(2,iter);
    end
    converged = sum(tol<opts.tol) > 10;

    % display progress every 1 second
    if iter==1 || toc(t(1)) > 1 || converged || iter==opts.maxit
        if iter==1
            display(W,f,noise_floor,ksp,iter,nrm,mask,opts); t(1:2) = tic();
        elseif t(1)==t(2)
            fprintf('Iterations per second: %.2f\n',(iter-1) / toc(t(1)));
            display(W,f,noise_floor,ksp,iter,nrm,mask,opts); t(1) = tic();
        else
            display(W,f,noise_floor,ksp,iter,nrm,mask,opts); t(1) = tic();
        end
    end

    % finish
    if converged || opts.maxit<=1; break; end

end

fprintf('Total time: %.1f sec (%i iters)\n',toc(t(2)),iter);

% remove 2x oversampling
if opts.osf > 1
    ok = nx/opts.osf/2+(1:nx/opts.osf);
    ksp = fftshift(ifft(ksp,[],1));
    ksp = ksp(ok,:,:,:,:);
    ksp = fft(ifftshift(ksp),[],1);

    basic = fftshift(ifft(basic,[],1));
    basic = basic(ok,:,:,:,:);
    basic = fft(ifftshift(basic),[],1);
end

% restore original orientation
if opts.readout==2
    ksp = permute(ksp,[2 1 3 4 5]);
    basic = permute(basic,[2 1 3 4]);
end

% only return first/last nrm
nrm = nrm(:,[1 end]);

% avoid dumping to screen
if nargout==0; clear; end

%% make data matrix
function [A opts] = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nc = size(data,3);
ne = size(data,4);
nh = size(data,5);
nk = opts.dims(6);

% precompute the circshifts with fast indexing
if ~isfield(opts,'ix')
    opts.ix = repmat(1:uint32(nx*ny*nc*ne*nh),[1 nk]);
    opts.ix = reshape(opts.ix,[nx ny nc ne nh nk]);
    for k = 1:nk
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        opts.ix(:,:,:,:,:,k) = circshift(opts.ix(:,:,:,:,:,k),[x y]);
    end
    if isa(data,'gpuArray'); opts.ix = gpuArray(opts.ix); end
end
A = data(opts.ix);

if opts.loraks
    A = cat(7,A,conj(A(opts.flip.x,opts.flip.y,:,:,:,:)));
end

A = reshape(A,nx*ny,[]);

%% undo data matrix
function [data opts] = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
ne = opts.dims(4);
nh = opts.dims(5);
nk = opts.dims(6);

A = reshape(A,nx,ny,nc,ne,nh,nk,[]);

if opts.loraks
    A(opts.flip.x,opts.flip.y,:,:,:,:,2) = conj(A(:,:,:,:,:,:,2));
end

% precompute the circshifts with fast indexing
if ~isfield(opts,'xi')
    opts.xi = reshape(1:uint32(numel(A)),size(A));
    for k = 1:nk
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        opts.xi(:,:,:,:,:,k,:) = circshift(opts.xi(:,:,:,:,:,k,:),-[x y]);
    end
    if isa(A,'gpuArray'); opts.xi = gpuArray(opts.xi); end
end
A = A(opts.xi);

data = mean(reshape(A,nx,ny,nc,ne,nh,[]),6);

%% off resonance + phase penalty function
function [nrm grd] = myfun(freq,data,opts)

nx = opts.dims(1);

% off resonance correction
roll = exp(i*opts.kx*freq(1));
data(:,:,:,:,1) = data(:,:,:,:,1)./roll;
data(:,:,:,:,2) = data(:,:,:,:,2).*roll;

% phase correction (not necessary but why not?)
r = dot(data(:,:,:,:,1),data(:,:,:,:,2));
d = dot(data(:,:,:,:,1),data(:,:,:,:,1));
r = reshape(r,[],1); d = reshape(d,[],1);
phi = angle((r'*d) / (d'*d)) / 2;

data(:,:,:,:,1) = data(:,:,:,:,1)./exp(i*phi);
data(:,:,:,:,2) = data(:,:,:,:,2).*exp(i*phi);

% for nuclear norm
A = make_data_matrix(data,opts);

% gradient
if nargout<2
    if size(A,1)<=size(A,2)
        W = svd(A,0);
    else
        W = svd(A'*A);
        W = sqrt(W);
    end
    dW = [];
else
    if size(A,1)<=size(A,2)
        [~,W,V] = svd(A,0);
        W = diag(W);
        V = V(:,1:numel(W));
    else
        [V W] = svd(A'*A);
        W = sqrt(diag(W));
    end
    dA = A.*opts.P;
    dW = real(diag(V'*(A'*dA)*V))./W;
end

% plain doubles for fminunc
nrm = gather(sum( W,'double'));
grd = gather(sum(dW,'double'));

%% show plots of various things
function display(W,f,noise_floor,ksp,iter,nrm,mask,opts)

nx = opts.dims(1);
ne = opts.dims(4);
nh = opts.dims(5);

% plot singular values
subplot(2,4,1); plot(W/W(1)); title(sprintf('rank %i/%i',nnz(f),numel(f)));
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]); ylim([0 1]); grid on;
line(xlim,min(1,gather([1 1]*noise_floor/W(1))),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% plot change in metrics
subplot(2,4,5);
if iter==1 && isfield(opts,'nrm') && opts.dims(5)>1 % only if nh>1
    plot(opts.range,opts.nrm,'-o'); title('off-resonance'); yticklabels([]);
    axis tight; ylabel('||A||_*','fontweight','bold'); xlabel('freq (deg/dwell)');
    line([1 1]*opts.freq,ylim,'linestyle','--','color','red'); grid on;
else
    ax = plotyy(1:iter,nrm(1,:),1:iter,nrm(2,:),'semilogy','semilogy');
    legend('||A||_*','||A||_F','location','northeast'); xlabel('iters');
    title(sprintf('tol %.2e',nrm(2,end))); axis(ax,'tight');
end

% mask on iter=1 to show the blackness of kspace
if iter==1; ksp = bsxfun(@times,ksp,mask); end

% prefer ims over imagesc
if exist('ims','file'); imagesc = @(x)ims(x,-0.99); end

% show current kspace (lines show center)
subplot(2,4,2); imagesc(log(sum(abs(ksp(:,:,:,1,1)),3)));
xlabel(num2str(size(ksp,2),'ky [%i]'));
ylabel(num2str(size(ksp,1),'kx [%i]'));
if nh==2; title('kspace (fwd)'); else; title('kspace (echo 1)'); end
line(xlim,[opts.center(1) opts.center(1)]);
line([opts.center(2) opts.center(2)],ylim);
if nh==2
    subplot(2,4,6); imagesc(log(sum(abs(ksp(:,:,:,1,2)),3)));
    xlabel(num2str(size(ksp,2),'ky [%i]'));
    ylabel(num2str(size(ksp,1),'kx [%i]'));
    title('kspace (rev)');
    line(xlim,[opts.center(1) opts.center(1)]);
    line([opts.center(2) opts.center(2)],ylim);
elseif ne>1
    subplot(2,4,6); imagesc(log(sum(abs(ksp(:,:,:,ne,1)),3)));
    xlabel(num2str(size(ksp,2),'ky [%i]'));
    ylabel(num2str(size(ksp,1),'kx [%i]'));
    title(sprintf('kspace (echo %i)',ne));
    line(xlim,[opts.center(1) opts.center(1)]);
    line([opts.center(2) opts.center(2)],ylim);
else
    subplot(2,4,6); imagesc(0); axis off;
end

% switch to image domain
ksp = fftshift(ifft2(ifftshift(ksp)));

% remove oversampling
if opts.osf > 1
    nx = size(ksp,1);
    ok = (nx/opts.osf/2)+(1:nx/opts.osf);
    ksp = ksp(ok,:,:,:,:);
end

% show current image
subplot(2,4,3); imagesc(sum(abs(ksp(:,:,:,1,1)),3));
xlabel(num2str(size(ksp,2),'y [%i]'));
ylabel(num2str(size(ksp,1),'x [%i]'));
if nh==2; title(sprintf('iter %i (fwd)',iter)); else; title(sprintf('iter %i (echo 1)',iter)); end
if nh==2
    subplot(2,4,7); imagesc(sum(abs(ksp(:,:,:,1,2)),3));
    xlabel(num2str(size(ksp,2),'y [%i]'));
    ylabel(num2str(size(ksp,1),'x [%i]'));
    title(sprintf('iter %i (rev)',iter));
elseif ne>1
    subplot(2,4,7); imagesc(sum(abs(ksp(:,:,:,ne,1)),3));
    xlabel(num2str(size(ksp,2),'y [%i]'));
    ylabel(num2str(size(ksp,1),'x [%i]'));
    title(sprintf('iter %i (echo %i)',iter,ne));
else
    subplot(2,4,7); imagesc(0); axis off;
end

% show one coil image phase
subplot(2,4,4); imagesc(angle(ksp(:,:,1,1,1)));
xlabel(num2str(size(ksp,2),'y [%i]'));
ylabel(num2str(size(ksp,1),'x [%i]'));
if nh==2; title(sprintf('phase (fwd)')); else; title(sprintf('phase (echo %i)',1)); end
if nh==2
    subplot(2,4,8); imagesc(angle(ksp(:,:,1,1,2)));
    xlabel(num2str(size(ksp,2),'y [%i]'));
    ylabel(num2str(size(ksp,1),'x [%i]'));
    title(sprintf('phase (rev)'));
elseif ne>1
    subplot(2,4,8); imagesc(angle(ksp(:,:,1,ne,1)));
    xlabel(num2str(size(ksp,2),'y [%i]'));
    ylabel(num2str(size(ksp,1),'x [%i]'));
    title(sprintf('phase (echo %i)',ne));
else
    subplot(2,4,8); imagesc(0); axis off;
end
drawnow;
