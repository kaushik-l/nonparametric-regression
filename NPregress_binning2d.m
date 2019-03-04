function [x,f] = NPregress_binning2d(xt,yt,dt,nbins)

% NPREGRESS_BINNING Performs nonparametric regression by binning
%   [x,f,pval] = NPregress_binning(xt,yt,binedges,nbootstraps,dt) performs 
% nonparametric regression of 'yt' onto 'xt' by grouping values in 'xt' 
% into bins specified by 'binedges'.
%
% 'nbootstraps' specifies the number of bootstrap repetitions used to 
% compute standard error of the mean of the estimator.
if nargin<4, nbins = []; end
if nargin<3, dt = []; end

if isempty(dt), dt = 1; end
if isempty(nbins), nbins = [10; 10]; end
nbins1 = nbins(1); nbins2 = nbins(2);
xt1 = xt(:,1); xt2 = xt(:,2);
binedges1 = linspace(min(xt1),max(xt1),nbins1+1);
binedges2 = linspace(min(xt2),max(xt2),nbins2+1);
bincntrs1 = 0.5*(binedges1(1:end-1) + binedges1(2:end));
bincntrs2 = 0.5*(binedges2(1:end-1) + binedges2(2:end));

%% determine tuning function
fval = cell(nbins1,nbins2);
for i=1:nbins1
    for j=1:nbins2
        indx = xt1>binedges1(i) & xt1<binedges1(i+1) & xt2>binedges2(j) & xt2<binedges2(j+1);
        fval{i,j} = yt(indx)/dt;
    end
end
x.mu = [bincntrs1 ; bincntrs2];
f.mu = cellfun(@nanmean,fval);