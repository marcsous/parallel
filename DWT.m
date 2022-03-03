classdef DWT
    
    % wavelet transform without the agonizing pain
    %
    % basically it takes care of the junk behind the
    % scenes so that dwt/idwt is as easy as fft/ifft
    %
    % note:
    % -always assumes periodic boundary conditions
    % -lax about input shape (vectorized ok) for pcg
    %
    % Example
    %   x = rand(1,8,'single');
    %   W = DWT(size(x),'db2');
    %   y = W * x; % forward
    %   z = W'* y; % inverse
    %   norm(x-z)
    %     ans = 1.0431e-07
    %
    %  -to threshold use W.thresh(x,sparsity) to do
    %   soft-thresholding to a given sparsity
   
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
                error('sizeINI is required');
            end
            if nargin<2
                wname = 'db2'; % orthogonal
            end
            if ~isnumeric(sizeINI) || ~isvector(sizeINI)
                error('sizeINI must be the output of size()');
            end
            while numel(sizeINI) && sizeINI(end)==1
                sizeINI(end) = []; % remove trailing ones
            end
            if any(sizeINI==0) || numel(sizeINI)>3
                error('only 1d, 2d or 3d supported');
            end
            if numel(sizeINI)<=2 && mod(max(sizeINI),2)
                error('only even dimensions supported');
            elseif numel(sizeINI)>2 && any(mod(sizeINI,2))
                error('only even dimensions supported');
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
            nc = numel(x) / prod(obj.sizeINI);
            if mod(nc,1)
                error('size(x) not compatible with sizeINI [%s]',num2str(obj.sizeINI));
            end

            % make it respect class
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
                
                y = reshape(x,prod(obj.sizeINI),nc);
                
                for c = 1:nc
                    y(:,c) = reshape(obj * y(:,c),[],1);
                end
                
                y = reshape(y,size(x)); % original size
                
            else
                
                % correct shape
                x = reshape(x,[obj.sizeINI 1]);
                
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
                        error('only 1d, 2d or 3d supported');
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
                        error('only 1d, 2d or 3d supported');
                    end
                    
                end
                
            end
            
        end
        
        %% threshold small coefficients
        function y = thresh(obj,x,sparsity)
            
            if ~exist('sparsity','var')
                sparsity = 0.5; % default
            elseif ~isscalar(sparsity) || ~isreal(sparsity) || sparsity<0 || sparsity>1
                error('sparsity must be a scalar between 0 and 1')
            end
            
            y = obj * x;

            % no. coils
            nc = numel(x) / prod(obj.sizeINI);
            
            % separate coils            
            y = reshape(y,[],nc);

            % soft-threshold to a target sparsity
            tmp = abs(y);
            [~,ok] = sort(tmp,'descend');
            k = ceil(size(y,1) * sparsity);

            for c = 1:nc
                %y(ok(k:end,c),c) = 0; % hard threshold
                thresh = tmp(ok(k,c),c); % soft threshold
                y(:,c) = max(0,tmp(:,c)-thresh).*sign(y(:,c));
            end
            
            y = obj' * y;
            
            y = reshape(y,size(x)); % original size
            
        end

        %% detect W' and set flag
        function obj = ctranspose(obj)
            
            obj.trans = ~obj.trans;
            
        end
        
    end
    
end

%% modified version of cell2mat to allow gpuArray

function m = cell2mat(c)
%CELL2MAT Convert the contents of a cell array into a single matrix.
%   M = CELL2MAT(C) converts a multidimensional cell array with contents of
%   the same data type into a single matrix. The contents of C must be able
%   to concatenate into a hyperrectangle. Moreover, for each pair of
%   neighboring cells, the dimensions of the cell's contents must match,
%   excluding the dimension in which the cells are neighbors. This constraint
%   must hold true for neighboring cells along all of the cell array's
%   dimensions.
%
%   The dimensionality of M, i.e. the number of dimensions of M, will match
%   the highest dimensionality contained in the cell array.
%
%   CELL2MAT is not supported for cell arrays containing cell arrays or
%   objects.
%
%	Example:
%	   C = {[1] [2 3 4]; [5; 9] [6 7 8; 10 11 12]};
%	   M = cell2mat(C)
%
%	See also MAT2CELL, NUM2CELL

% Copyright 1984-2010 The MathWorks, Inc.

% Error out if there is no input argument
if nargin==0
    error(message('MATLAB:cell2mat:NoInputs'));
end
% short circuit for simplest case
elements = numel(c);
if elements == 0
    m = [];
    return
end
if elements == 1
    if isnumeric(c{1}) || ischar(c{1}) || islogical(c{1}) || isstruct(c{1})
        m = c{1};
        return
    end
end
% Error out if cell array contains mixed data types
cellclass = class(c{1});
ciscellclass = cellfun('isclass',c,cellclass);
if ~all(ciscellclass(:))
    error(message('MATLAB:cell2mat:MixedDataTypes'));
end

% Error out if cell array contains any cell arrays or objects
ciscell = iscell(c{1});
if isa(c{1},'gpuArray')
    cisobj = isobject(classUnderlying(c{1}));
else
    cisobj = isobject(c{1});
end
if cisobj || ciscell
    error(message('MATLAB:cell2mat:UnsupportedCellContent'));
end

% If cell array of structures, make sure the field names are all the same
if isstruct(c{1})
    cfields = cell(elements,1);
    for n=1:elements
        cfields{n} = fieldnames(c{n});
    end
    % Perform the actual field name equality test
    if ~isequal(cfields{:})
        error(message('MATLAB:cell2mat:InconsistentFieldNames'));
    end
end

% If cell array is 2-D, execute 2-D code for speed efficiency
if ndims(c) == 2
    rows = size(c,1);
    cols = size(c,2);   
    if (rows < cols)
        m = cell(rows,1);
        % Concatenate one dim first
        for n=1:rows
            m{n} = cat(2,c{n,:});
        end
        % Now concatenate the single column of cells into a matrix
        m = cat(1,m{:});
    else
        m = cell(1, cols);
        % Concatenate one dim first
        for n=1:cols
            m{n} = cat(1,c{:,n});
        end    
        % Now concatenate the single column of cells into a matrix
        m = cat(2,m{:});
    end
    return
end

csize = size(c);
% Treat 3+ dimension arrays

% Construct the matrix by concatenating each dimension of the cell array into
%   a temporary cell array, CT
% The exterior loop iterates one time less than the number of dimensions,
%   and the final dimension (dimension 1) concatenation occurs after the loops

% Loop through the cell array dimensions in reverse order to perform the
%   sequential concatenations
for cdim=(length(csize)-1):-1:1
    % Pre-calculated outside the next loop for efficiency
    ct = cell([csize(1:cdim) 1]);
    cts = size(ct);
    ctsl = length(cts);
    mref = {};

    % Concatenate the dimension, (CDIM+1), at each element in the temporary cell
    %   array, CT
    for mind=1:prod(cts)
        [mref{1:ctsl}] = ind2sub(cts,mind);
        % Treat a size [N 1] array as size [N], since this is how the indices
        %   are found to calculate CT
        if ctsl==2 && cts(2)==1
            mref = {mref{1}};
        end
        % Perform the concatenation along the (CDIM+1) dimension
        ct{mref{:}} = cat(cdim+1,c{mref{:},:});
    end
    % Replace M with the new temporarily concatenated cell array, CT
    c = ct;
end

% Finally, concatenate the final rows of cells into a matrix
m = cat(1,c{:});

end