function [singvals sigma] = optimal_shrinkage(singvals,beta,loss,sigma)

% function singvals = optimal_shrinkage(singvals,beta,sigma_known)
%
% Perform optimal shrinkage (w.r.t one of a few possible losses) on data
% singular values, when the noise is assumed white, and the noise level is known 
% or unknown.
%
% IN:
%   singvals: a vector of data singular values, obtained by running svd
%             on the data matrix
%   beta:     aspect ratin m/n of the m-by-n matrix whose singular values 
%             are given
%   loss:     the loss function for which the shrinkage should be optimal
%             presently implmented: 'fro' (Frobenius or square Frobenius norm loss = MSE)        
%                                   'nuc' (nuclear norm loss)
%                                   'op'  (operator norm loss)
%   sigma:    (optional) noise standard deviation (of each entry of the noise matrix) 
%             if this argument is not provided, the noise level is estimated 
%             from the data.
%
% OUT:
%    singvals: the vector of singular values after performing optimal shrinkage
%
% Usage:
%   Given an m-by-n matrix Y known to be low rank and observed in white noise
%   with zero mean, form a denoied matrix Xhat by:
%   
%   [U D V] = svd(Y,'econ');
%   y = diag(Y);
%   y = optimal_shrinkage(y,m/n,'fro');
%   Xhat = U * diag(y) * V';
%
%   where you can replace 'fro' with one of the other losses. 
%   if the noise level sigma is known, in the third line use instead
%       y = optimal_shrinkage(y,m/n,'fro',sigma);
%    
% -----------------------------------------------------------------------------
% Authors: Matan Gavish and David Donoho <lastname>@stanford.edu, 2013
% 
% This program is free software: you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
% details.
%
% You should have received a copy of the GNU General Public License along with
% this program.  If not, see <http://www.gnu.org/licenses/>.
% -----------------------------------------------------------------------------
%
%
% 2018 MB: NOTE THIS WAS MODIFIED TO RETURN SIGMA AND REDUCE ROUNDOFF ERRORS.
%
%
assert(prod(size(beta))==1)
assert(beta<=1)
assert(beta>0)
assert(prod(size(singvals))==length(singvals))
assert(ismember(loss,{'fro','op','nuc'}))

% estimate sigma if needed
if nargin<4 || isempty(sigma)
    warning('off','MATLAB:quadl:MinStepSize')
    MPmedian = MedianMarcenkoPastur(beta);
    sigma = median(singvals) / sqrt(MPmedian);
    fprintf('estimated sigma=%0.2e \n',sigma);
end

singvals = optshrink_impl(singvals,beta,loss,sigma);

end

function singvals = optshrink_impl(singvals,beta,loss,sigma)

    %y = @(x)( (1+sqrt(beta)).*(x<=beta^0.25) + sqrt((x+1./x) ...
    %     .* (x+beta./x)).*(x>(beta^0.25)) );
    assert(sigma>0)
    assert(prod(size(sigma))==1)

    sqrt0 = @(arg)sqrt(max(arg,0)); % MB no sqrt negatives
    x = @(y)( sqrt0(0.5*((y.^2-beta-1 )+sqrt0((y.^2-beta-1).^2 - 4*beta) ))...
        .* (y>=1+sqrt(beta)));
    
    
    opt_fro_shrink = @(y)( sqrt0(((y.^2-beta-1).^2 - 4*beta) ) ./ y);
    opt_op_shrink = @(y)(max(x(y),0));
    opt_nuc_shrink = @(y)(max(0, (x(y).^4 - sqrt(beta)*x(y).*y - beta)) ...
        ./((x(y).^2) .* y));
   
    y = singvals/sigma;
    
    switch loss
    case 'fro'
        singvals = sigma * opt_fro_shrink(y);
    case 'nuc'
        singvals = sigma * opt_nuc_shrink(y);
        singvals((x(y).^4 - sqrt(beta)*x(y).*y - beta)<=0)=0;
    case 'op'
        singvals = sigma * opt_op_shrink(y);
    otherwise
        error('loss unknown')
    end

    % MB handle roundoff errors
    for k = 2:numel(singvals)
        if singvals(k)>singvals(k-1)
            singvals(k) = 0;
        end
    end
    
end


function I = MarcenkoPasturIntegral(x,beta)
    if beta <= 0 | beta > 1,
        error('beta beyond')
    end
    lobnd = (1 - sqrt(beta))^2;
    hibnd = (1 + sqrt(beta))^2;
    if (x < lobnd) | (x > hibnd),
        error('x beyond')
    end
    dens = @(t) sqrt((hibnd-t).*(t-lobnd))./(2*pi*beta.*t);
    I = quadl(dens,lobnd,x);
    fprintf('x=%.3f,beta=%.3f,I=%.3f\n',x,beta,I);
end


function med = MedianMarcenkoPastur(beta)
    MarPas = @(x) 1-incMarPas(x,beta,0);
    lobnd = (1 - sqrt(beta))^2;
    hibnd = (1 + sqrt(beta))^2;
    change = 1;
    while change & (hibnd - lobnd > .001),
      change = 0;
      x = linspace(lobnd,hibnd,5);
      for i=1:length(x),
          y(i) = MarPas(x(i));
      end
      if any(y < 0.5),
         lobnd = max(x(y < 0.5));
         change = 1;
      end
      if any(y > 0.5),
         hibnd = min(x(y > 0.5));
         change = 1;
      end
    end
    med = (hibnd+lobnd)./2;
end

function I = incMarPas(x0,beta,gamma)
    if beta > 1,
        error('betaBeyond');
    end
    topSpec = (1 + sqrt(beta))^2;
    botSpec = (1 - sqrt(beta))^2;
    MarPas = @(x) IfElse((topSpec-x).*(x-botSpec) >0, ...
                         sqrt((topSpec-x).*(x-botSpec))./(beta.* x)./(2 .* pi), ...
                         0);
    if gamma ~= 0,
       fun = @(x) (x.^gamma .* MarPas(x));
    else
       fun = @(x) MarPas(x);
    end
    I = quadl(fun,x0,topSpec);
    
    function y=IfElse(Q,point,counterPoint)
        y = point;
        if any(~Q),
            if length(counterPoint) == 1,
                counterPoint = ones(size(Q)).*counterPoint;
            end
            y(~Q) = counterPoint(~Q);
        end
        
    end
end


