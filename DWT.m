classdef DWT
    % Q = DWT(sizeINI,wname) ['db1','db2','db3']
    %
    % wavelet transform without the agonizing pain.
    % basically takes care of the junk behind the
    % scenes so dwt/idwt is as easy as fft/ifft.
    %
    % notes:
    % -always assumes periodic boundary conditions
    % -relaxed about input shape (vectorized ok)
    % -accepts leading or trailing coil dimensions
    % -Q.thresh(x,sparsity) does soft thresholding
    %  to a given sparsity (e.g. 0.25 => 25% zeros)
    %
    %% Example:
    %   x = (1:8) + 0.1*randn(1,8);
    %   Q = DWT(size(x),'db2');
    %   y = Q * x; % forward
    %   z = Q'* y; % inverse
    %   norm(x-z,Inf)
    %     ans = 2.3801e-12
    %   z = Q.thresh(x,0.25); % 25% zeros
    %   norm(x-z,Inf)
    %     ans = 0.2362
    %   [x;z]
    %     ans =
    %       0.9468    1.9106    3.1145    4.0127    4.7686    6.0869    6.9622    8.1000
    %       0.8835    1.9429    2.9707    3.8405    4.7527    5.8872    6.9622    7.8637
    %   Q*[x;z]
    %     ans =
    %       4.7292    3.8104    6.3904   10.4568   -1.1663    0.1082    0.1412    3.9703
    %       4.5880    3.6692    6.2492   10.3156   -1.0251         0         0    3.8291
    %
    %% Bugfixes (for older matlab versions)
    % /usr/local/MATLAB/R2018b/toolbox/wavelet/wavelet/idwt3.m
    % < Z = zeros(sX(1),2*sX(2)-1,sX(3));
    % > Z = zeros(sX(1),2*sX(2)-1,sX(3),'like',X);
    %
    % /usr/local/MATLAB/R2018b/toolbox/wavelet/wavelet/dwt.m
    % < validateattributes(x,{'numeric'},{'vector','finite','real'},'dwt','X');
    % > validateattributes(x,{'numeric'},{'vector','finite'},'dwt','X');
    %
    % /usr/local/MATLAB/R2018b/toolbox/wavelet/wavelet/idwt.m
    % < validateattributes(...,{'numeric'},{'vector','finite','real'}...
    % > validateattributes(...,{'numeric'},{'vector','finite'}...
    %
    % /usr/local/MATLAB/R2018b/toolbox/matlab/datatypes/cell2mat.m
    % < cisobj = isobject(c{1});
    % > if isa(c{1},'gpuArray'); cisobj = ~isnumeric(c{1}); else; cisobj = isobject(c{1}); end
    %
    % /usr/local/MATLAB/R2018b/toolbox/wavelet/wavelet/dyadup.m
    % < zeros(...);
    % > zeros(...,'like',x);
    %
    % /usr/local/MATLAB/R2018b/toolbox/wavelet/wavelet/dyadup.m
    % < if r>1, y = y'; end
    % > if r>1, y = y.'; end
    %
    % /usr/local/MATLAB/R2018b/toolbox/wavelet/wavelet/wextend.m
    % < validateattributes(x,{'numeric'}...
    % > validateattributes(x,{'numeric','gpuArray'}...

    properties (SetAccess = private)
        sizeINI
        wname = 'db1'; % orthogonal
        mode = 'per'; % periodic
        trans = false % transpose
        filters
        dec
    end

    methods
        
        %% constructor
        function obj = DWT(sizeINI,wname)
            
            if nargin<1
                error('sizeINI is required.');
            elseif nargin==2
                obj.wname = wname;
            elseif nargin>2
                error('Wrong number of arguments.')
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
            
            [LoD HiD LoR HiR] = wfilters(obj.wname);
            obj.filters.LoD = LoD;
            obj.filters.HiD = HiD;
            obj.filters.LoR = LoR;
            obj.filters.HiR = HiR;
            
        end
        
        %% y = Q*x or y = Q'*x
        function y = mtimes(obj,x)
            
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
            
            % loop over extra dimensions (coils)
            [nc dim sx] = get_coils(obj,x);
            
            if nc>1

                % expand coil dimension
                y = reshape(x,sx);
                
                % indices for each dimension
                ix = repmat({':'},numel(sx),1);    
                
                % transform coils separately
                for c = 1:nc
                    ix{dim} = c;
                    y(ix{:}) = obj * y(ix{:});
                end

                % original shape
                y = reshape(y,size(x)); 
                
            else
                
                % correct shape
                y = reshape(x,obj.sizeINI);

                if obj.trans==0
                    
                    % forward transform
                    if isvector(y)
                        [CA CD] = dwt(y,LoD,HiD,'mode',obj.mode);
                        if isrow(y); y = [CA CD]; else; y = [CA;CD]; end
                    elseif ndims(y)==2
                        [CA CH CV CD] = dwt2(y,LoD,HiD,'mode',obj.mode);
                        y = [CA CV; CH CD];
                    elseif ndims(y)==3
                        wt = dwt3(y,{LoD,HiD,LoR,HiR},'mode',obj.mode);
                        y = cell2mat(wt.dec);
                    else
                        error('only 1d, 2d or 3d supported.');
                    end
                    
                else
                    
                    % inverse transform
                    if isvector(y)
                        if isrow(y)
                            C = mat2cell(y,1,[obj.sizeINI(2)/2 obj.sizeINI(2)/2]);
                        else
                            C = mat2cell(y,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],1);
                        end
                        y = idwt(C{1},C{2},LoR,HiR,'mode',obj.mode);
                    elseif ndims(y)==2
                        C = mat2cell(y,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],[obj.sizeINI(2)/2 obj.sizeINI(2)/2]);
                        y = idwt2(C{1},C{2},C{3},C{4},LoR,HiR,'mode',obj.mode);
                    elseif ndims(y)==3
                        wt.sizeINI = obj.sizeINI;
                        wt.filters.LoD = {LoD,LoD,LoD};
                        wt.filters.HiD = {HiD,HiD,HiD};
                        wt.filters.LoR = {LoR,LoR,LoR};
                        wt.filters.HiR = {HiR,HiR,HiR};
                        wt.mode = obj.mode;
                        wt.dec = mat2cell(y,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],[obj.sizeINI(2)/2 obj.sizeINI(2)/2],[obj.sizeINI(3)/2 obj.sizeINI(3)/2]);
                        y = idwt3(wt);
                    else
                        error('only 1d, 2d or 3d supported.');
                    end
                    
                end
                
                % original shape
                y = reshape(y,size(x));
                
            end
            
            %% get number of coils
            function [nc dim sx] = get_coils(obj,x)
                
                sx = size(x);

                % number of coils
                nc = prod(sx) / prod(obj.sizeINI);
                
                if mod(nc,1)
                    error('Expansion not compatible with sizeINI=[%s].',num2str(obj.sizeINI,'%i '));
                end  
                
                % get coil dimension
                dim = 0;
                for d = 1:numel(sx)
                    if sx(d)~=dimsize(obj,d)
                        dim = d;
                        break;
                    end
                end

                % expand array dimension, e.g. [2n] => [n 2]
                if dim>1 || iscolumn(x)
                    sx = [obj.sizeINI nc];
                    dim = numel(sx);
                end
                
            end
            
        end
        
        %% threshold wavelet coefficients
        function [y lambda] = thresh(obj,x,sparsity)
            
            if nargin<3 || ~isscalar(sparsity) || ~isreal(sparsity) || sparsity<0 || sparsity>1
                error('sparsity must be a scalar between 0 and 1.')
            end
            
            % to wavelet domain
            y = obj * x;

            % soft threshold all coils together
            y = reshape(y,[],1);

            absy = abs(y);
            signy = sign(y);
            
            v = sort(absy,'ascend');
            index = round(numel(v) * sparsity);
            
            if index==0
                lambda = cast(0,'like',v);
            else
                lambda = v(index);
                y = signy .* max(absy-lambda,0);
            end

            % to image domain
            y = obj' * y;
            
            % original shape
            y = reshape(y,size(x));

        end

        %% detect Q' and set flag
        function obj = ctranspose(obj)
            
            obj.trans = ~obj.trans;
            
        end
       
        %% dimension size
        function n = dimsize(obj,dim)
            
            if nargin<2
                n = obj.sizeINI;
            elseif ~isscalar(dim) || ~isnumeric(dim) || ~isreal(dim) || dim<1 || mod(dim,1)~=0
                error('Dimension argument must be a positive integer scalar within indexing range.');
            elseif dim <= numel(obj.sizeINI)
                n = obj.sizeINI(dim);
            else
                n = 1;
            end
            
        end
        
    end
    
end

