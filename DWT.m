classdef DWT
    % Q = DWT(sizeINI)
    %
    % Daubechies wavelet transform (db2)
    %
    % notes:
    % -always assumes periodic boundary conditions
    % -relaxed about input shape (vectorized ok)
    % -accepts leading or trailing coil dimensions
    % -Q.thresh(x,sparsity) does soft thresholding
    %  to a given sparsity (e.g. 0.25 => 25% zeros)
    %
    % Example:
    %{
       x = (1:8) + 0.1*randn(1,8);
       Q = DWT(size(x));
       y = Q * x; % forward
       z = Q'* y; % inverse
       norm(x-z,Inf)
         ans = 2.3801e-12
       z = Q.thresh(x,0.25); % 25% zeros
       norm(x-z,Inf)
         ans = 0.2362
       [x;z]
         ans =
           0.9468    1.9106    3.1145    4.0127    4.7686    6.0869    6.9622    8.1000
           0.8835    1.9429    2.9707    3.8405    4.7527    5.8872    6.9622    7.8637
       Q*[x;z]
         ans =
           4.7292    3.8104    6.3904   10.4568   -1.1663    0.1082    0.1412    3.9703
           4.5880    3.6692    6.2492   10.3156   -1.0251         0         0    3.8291
    %}
    properties (SetAccess = private)
        sizeINI
        trans = false
    end

    methods
        
        %% constructor
        function obj = DWT(sizeINI)
            
            if ~isnumeric(sizeINI) || ~isvector(sizeINI)
                error('sizeINI must be the output of size().');
            end
            while numel(sizeINI)>2 && sizeINI(end)==1
                sizeINI(end) = []; % remove trailing ones
            end
            if numel(sizeINI)==1
                sizeINI(end+1) = 1; % append a trailing one
            end
            if any(mod(sizeINI,2) & sizeINI~=1)
                error('only even dimensions supported.');
            end
            obj.sizeINI = reshape(sizeINI,1,[]);
            
        end
        
        %% y = Q*x or y = Q'*x
        function y = mtimes(obj,x)
            
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

                % convolutions
                LoD = [ 1-sqrt(3); 3-sqrt(3); 3+sqrt(3); 1+sqrt(3)] / sqrt(32);
                HiD = [-1-sqrt(3); 3+sqrt(3);-3+sqrt(3); 1-sqrt(3)] / sqrt(32);
                LoR = [ 1+sqrt(3); 3+sqrt(3); 3-sqrt(3); 1-sqrt(3)] / sqrt(32);
                HiR = [ 1-sqrt(3);-3+sqrt(3); 3+sqrt(3);-1-sqrt(3)] / sqrt(32);

                LoD = cast(LoD,'like',y); HiD = cast(HiD,'like',y);
                LoR = cast(LoR,'like',y); HiR = cast(HiR,'like',y);
                
                if obj.trans==0

                    % forward transform
                    for d = 1:numel(obj.sizeINI)
                        if obj.sizeINI(d) > 1

                            ylo = cconvn(y,LoD);
                            yhi = cconvn(y,HiD);

                            ix = repmat({':'},numel(obj.sizeINI),1);
                            ix{d} = 1:2:obj.sizeINI(d); % odd indices
     
                            ylo = ylo(ix{:});
                            yhi = yhi(ix{:});                         
                         
                            y = cat(d,ylo,yhi);
                            
                        end
                        LoD = reshape(LoD,[1 size(LoD)]);
                        HiD = reshape(HiD,[1 size(HiD)]); 
                    end
                    
                else
                   
                    % inverse transform
                    for d = 1:numel(obj.sizeINI)
                        if obj.sizeINI(d) > 1
                            
                            ix = repmat({':'},numel(obj.sizeINI),1);
                            lo = ix; lo{d} = 1:obj.sizeINI(d)/2;
                            hi = ix; hi{d} = 1+obj.sizeINI(d)/2:obj.sizeINI(d);
                            
                            ylo = y(lo{:});
                            yhi = y(hi{:});

                            ix = repmat({':'},numel(obj.sizeINI),1);
                            ix{d}  = 2:2:obj.sizeINI(d); % even indices
        
                            tmp = zeros(size(y),'like',y);
                            tmp(ix{:}) = ylo; ylo = tmp;
                            tmp(ix{:}) = yhi; yhi = tmp;

                            y = cconvn(ylo,LoR) + cconvn(yhi,HiR);
                        end
                        LoR = reshape(LoR,[1 size(LoR)]);
                        HiR = reshape(HiR,[1 size(HiR)]); 
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
            
            if nargin<3
                error('Not enough input arguments.');
            end
            if ~isscalar(sparsity) || ~isreal(sparsity) || sparsity<0 || sparsity>1
                error('sparsity must be a scalar between 0 and 1.')
            end
            
            % to wavelet domain
            y = obj * x;

            % soft threshold coils separately
            y = reshape(y,prod(obj.sizeINI),[]);

            absy = abs(y);
            signy = sign(y);
            
            v = sort(absy,'ascend');
            index = round(prod(obj.sizeINI) * sparsity);
            
            if index==0
                lambda = cast(0,'like',v);
            else
                lambda = v(index,:);
                y = signy .* max(absy-lambda,0);
            end

            % to image domain
            y = obj' * reshape(y,size(x));

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

