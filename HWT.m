classdef HWT
    % Q = HWT(sizeINI)
    %
    % Haar wavelet transform
    %
    % notes:
    % -always assumes periodic boundary conditions
    % -relaxed about input shape (vectorized ok)
    % -accepts leading or trailing coil dimensions
    % -Q.thresh(x,sparsity) does soft thresholding
    %  to a given sparsity (e.g. 0.25 => 25% zeros)

    properties (SetAccess = private)
        sizeINI
        trans = false
    end

    methods
        
        %% constructor
        function obj = HWT(sizeINI)
            
            if ~isnumeric(sizeINI) || ~isvector(sizeINI)
                error('sizeINI must be the output of size().');
            end
            while numel(sizeINI)>2 && sizeINI(end)==1
                sizeINI(end) = []; % remove trailing ones
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

                if obj.trans==0

                    % forward transform
                    for d = 1:numel(sx)
                        if sx(d) > 1
                            
                            ix = repmat({':'},numel(sx),1);
                            
                            odd  = ix; odd{d}  = 1:2:sx(d);
                            even = ix; even{d} = 2:2:sx(d);
                            
                            yodd  = y(odd{:});
                            yeven = y(even{:});
                            
                            y = cat(d,yodd+yeven,yodd-yeven) / sqrt(2);
                            
                        end
                    end
                    
                else
                    
                    % inverse transform
                    for d = 1:numel(sx)
                        if sx(d) > 1
                            
                            ix = repmat({':'},numel(sx),1);
                            
                            lo = ix; lo{d} = 1:sx(d)/2;
                            hi = ix; hi{d} = 1+sx(d)/2:sx(d);
                            
                            ylo = y(lo{:});
                            yhi = y(hi{:});
                            
                            ix = repmat({':'},numel(sx),1);
                            
                            odd  = ix; odd{d}  = 1:2:sx(d);
                            even = ix; even{d} = 2:2:sx(d);
                            
                            y(odd{:})  = (ylo+yhi) / sqrt(2);
                            y(even{:}) = (ylo-yhi) / sqrt(2);                       
                            
                        end
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

