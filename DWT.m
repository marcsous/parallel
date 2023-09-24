classdef DWT
    
    % wavelet transform without the agonizing pain.
    % basically takes care of the junk behind the
    % scenes so dwt/idwt is as easy as fft/ifft.
    %
    % notes:
    % -always assumes periodic boundary conditions
    % -relaxed about input shape (pcg vectorized ok) 
    % -Q.thresh(x,sparsity) does soft thresholding
    %  to a given sparsity (e.g. 0.25 => 25% zeros)
    %
    % BUGFIX 1 (for older matlab versions)
    % /usr/local/MATLAB/R2018b/toolbox/wavelet/wavelet/idwt3.m
    % < Z = zeros(sX(1),2*sX(2)-1,sX(3));
    % > Z = zeros(sX(1),2*sX(2)-1,sX(3),'like',X);
    %
    % BUGFIX 2 (for older matlab versions)
    % /usr/local/MATLAB/R2018b/toolbox/matlab/datatypes/cell2mat.m
    % < cisobj = isobject(c{1});
    % > if isa(c{1},'gpuArray'); cisobj = ~isnumeric(c{1}); else; cisobj = isobject(c{1}); end
    %
    % Example:
    %   x = (1:8) + 0.1*randn(1,8);
    %   Q = DWT(size(x),'db2');
    %   y = Q * x; % forward
    %   z = Q'* y; % inverse
    %   norm(x-z,Inf)
    %     ans = 2.4203e-12
    %   z = Q.thresh(x,0.25); % 25% zeros
    %   norm(x-z,Inf)
    %     ans = 0.1426
    %   [x;z]
    %     ans =
    %       1.0490    2.0739    3.1712    3.9806    4.7862    5.9160    7.1355    7.8928
    %       1.0108    2.0900    3.0785    3.8766    4.7257    5.9381    7.0532    7.7502
    %   Q * [x;z]
    %     ans =
    %       4.7439    3.9272    6.3288   10.4595   -1.0532    0.0391   -0.0852    3.7309
    %       4.6587    3.8420    6.2436   10.3743   -0.9680         0         0    3.6456
       
    properties (SetAccess = private)
        sizeINI
        filters
        dec
        mode = 'per'; % must be periodic
        trans = 0 % transpose flag (0 or 1)
    end

    methods
        
        %% constructor
        function obj = DWT(sizeINI,wname)
            
            if nargin<1
                error('sizeINI is required.');
            end
            if nargin<2
                wname = 'db2'; % orthogonal
            end
            if ~isnumeric(sizeINI) || ~isvector(sizeINI)
                error('sizeINI must be the output of size().');
            end
            while numel(sizeINI)>2 && sizeINI(end)==1
                sizeINI(end) = []; % remove trailing ones
            end
            if any(sizeINI==0) || numel(sizeINI)>3
                error('only 1d, 2d or 3d supported.');
            end
            if any(sizeINI>1 & mod(sizeINI,2))
                error('only even dimensions supported.');
            end
            obj.sizeINI = reshape(sizeINI,1,[]);
            
            [LoD HiD LoR HiR] = wfilters(wname);
            obj.filters.LoD = LoD;
            obj.filters.HiD = HiD;
            obj.filters.LoR = LoR;
            obj.filters.HiR = HiR;
            
        end
        
        %% y = W*x or y = W'*x
        function y = mtimes(obj,x)
            
            % number of coils
            [nc dim sz] = get_coils(obj,x);
            
            % make dwt respect class
            LoD = obj.filters.LoD;
            HiD = obj.filters.HiD;
            LoR = obj.filters.LoR;
            HiR = obj.filters.HiR;
            if isa(x,'single') || (isa(x,'gpuArray') && isequal(classUnderlying(x),'single'))
                LoD = single(LoD);
                HiD = single(HiD);
                LoR = single(LoR);
                HiR = single(HiR);
            end
            
            % allow looping over coil dimension            
            if nc>1
                
                % expand coil dimension
                y = reshape(x,sz);
                
                % indices for the expanded array
                ix = cell(size(sz));
                for d = 1:numel(ix); ix{d} = ':'; end
                
                % transform one coil at a time
                for c = 1:nc
                    ix{dim} = c;
                    y(ix{:}) = obj * reshape(y(ix{:}),obj.sizeINI);
                end
                
                y = reshape(y,size(x)); % original size
                
            else
                
                % correct shape
                x = reshape(x,obj.sizeINI);

                if obj.trans==0
                    
                    % forward transform
                    if isvector(x)
                        [CA CD] = dwt(x,LoD,HiD,'mode',obj.mode);
                        if isrow(x); y = [CA CD]; else; y = [CA;CD]; end
                    elseif ndims(x)==2
                        [CA CH CV CD] = dwt2(x,LoD,HiD,'mode',obj.mode);
                        y = [CA CV; CH CD];
                    elseif ndims(x)==3
                        wt = dwt3(x,{LoD,HiD,LoR,HiR},'mode',obj.mode);
                        y = cell2mat(wt.dec);
                    else
                        error('only 1d, 2d or 3d supported.');
                    end
                    
                else
                    
                    % inverse transform
                    if isvector(x)
                        if isrow(x)
                            C = mat2cell(x,1,[obj.sizeINI(2)/2 obj.sizeINI(2)/2]);
                        else
                            C = mat2cell(x,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],1);
                        end
                        y = idwt(C{1},C{2},LoR,HiR,'mode',obj.mode);
                    elseif ndims(x)==2
                        C = mat2cell(x,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],[obj.sizeINI(2)/2 obj.sizeINI(2)/2]);
                        y = idwt2(C{1},C{2},C{3},C{4},LoR,HiR,'mode',obj.mode);
                    elseif ndims(x)==3
                        wt.sizeINI = obj.sizeINI;
                        wt.filters.LoD = {LoD,LoD,LoD};
                        wt.filters.HiD = {HiD,HiD,HiD};
                        wt.filters.LoR = {LoR,LoR,LoR};
                        wt.filters.HiR = {HiR,HiR,HiR};
                        wt.mode = obj.mode;
                        wt.dec = mat2cell(x,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],[obj.sizeINI(2)/2 obj.sizeINI(2)/2],[obj.sizeINI(3)/2 obj.sizeINI(3)/2]);
                        y = idwt3(wt);
                    else
                        error('only 1d, 2d or 3d supported.');
                    end
                    
                end
                
            end
            
        end
        
        %% threshold small coefficients
        function y = thresh(obj,x,sparsity)
            
            if ~exist('sparsity','var')
                sparsity = 0.25; % default
            elseif ~isscalar(sparsity) || ~isreal(sparsity) || ~isnumeric(sparsity) || sparsity<0 || sparsity>1
                error('sparsity must be a scalar between 0 and 1.')
            end
            
            % to wavelet domain
            y = obj * x;

            % number of coils
            [nc dim sz] = get_coils(obj,x);
            
            % expand coil dimension            
            y = reshape(y,sz);

            % indices for the expanded array
            ix = cell(size(sz));
            for d = 1:numel(ix); ix{d} = ':'; end
                
            % threshold one coil at a time
            for c = 1:nc
                
                if dim; ix{dim} = c; end
                absy = abs(y(ix{:}));
                signy = sign(y(ix{:}));
                
                [~,k] = sort(absy(:),'ascend');
                maxk = round(numel(k) * sparsity);
                
                thresh = absy(k(maxk)); % soft threshold
                y(ix{:}) = max(0,absy-thresh).*signy;

            end

            y = reshape(y,size(x)); % original size

            % to image domain
            y = obj' * y;
            
            y = reshape(y,size(x)); % original size
            
        end

        %% detect W' and set flag
        function obj = ctranspose(obj)
            
            obj.trans = ~obj.trans;
            
        end
        
        %% get number of coils
        function [nc dim sz] = get_coils(obj,x)
            
            % number of coils
            nc = numel(x) / prod(obj.sizeINI);
            
            % coil dimension
            dim = 0;
            for d = 1:max(ndims(x),numel(obj.sizeINI))
                if d>numel(obj.sizeINI)
                    sizeINI_d = 1;
                else
                    sizeINI_d = obj.sizeINI(d);
                end
                
                if size(x,d)~=sizeINI_d
                    if dim==0
                        dim = d;
                    else
                        dim = -1; % error: more than one dim
                    end
                end
            end
            
            % catch problems
            if mod(nc,1) || dim<0
                error('size(x)=[%s] not compatible with sizeINI=[%s].',num2str(size(x),'%i '),num2str(obj.sizeINI,'%i '));
            end
            
            % expanded array dimensions
            if dim==0 || size(x,dim)==nc
                sz = size(x);
            else
                sz = [obj.sizeINI(1:dim) obj.sizeINI(dim:end)];
                dim = dim+1;
                sz(dim) = nc;
            end

        end
        
    end
    
end

