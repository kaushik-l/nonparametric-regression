function [x,f,pval] = NPregress_binning(xt,yt,binedges,nbootstraps,dt)

% NPREGRESS_BINNING Performs nonparametric regression by binning
%   [x,f,pval] = NPregress_binning(xt,yt,binedges,nbootstraps,dt) performs 
% nonparametric regression of 'yt' onto 'xt' by grouping values in 'xt' 
% into bins specified by 'binedges'.
%
% 'nbootstraps' specifies the number of bootstrap repetitions used to 
% compute standard error of the mean of the estimator.
if nargin<4, nbootstraps = []; dt = 1;
elseif nargin<5, dt = 1; end
if isempty(dt), dt = 1; end
if ~isempty(nbootstraps), compute_sem = 1; end
nbins = length(binedges)-1;

%% test statistical significance of tuning
xval = cell(nbins,1);
xgroup = cell(nbins,1);
fval = cell(nbins,1);
for i=1:nbins
    indx = xt>binedges(i) & xt<binedges(i+1);
    xval{i} = xt(indx);
    xgroup{i} = cell(length(xval{i}),1); xgroup{i}(:) = {num2str(i)};
    fval{i} = yt(indx)/dt;
end
if nargout>2
    pval = anova1(cell2mat(fval),vertcat(xgroup{:}),'off'); % one-way unbalanced anova
end

%% determine tuning function
if ~compute_sem % just return the means
    x.mu = cellfun(@mean,xval);
    f.mu = cellfun(@mean,fval);
else % obtain both mean and sem by bootstrapping (slow)
    x_mu = zeros(nbootstraps,nbins);
    f_mu = zeros(nbootstraps,nbins);
    for i=1:nbins
        indx = find(xt>binedges(i) & xt<binedges(i+1));
        for j=1:nbootstraps
            sampindx = randsample(indx,length(indx),true); % sample with replacement
            x_mu(j,i) = mean(xt(sampindx));
            f_mu(j,i) = mean(yt(sampindx)/dt);
        end
    end
    x.mu = mean(x_mu);
    x.sem = std(x_mu);
    f.mu = mean(f_mu); % mean
    f.sem = std(f_mu); % standard error of the mean
end