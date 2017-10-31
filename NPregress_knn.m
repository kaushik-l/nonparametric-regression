function [x,f] = NPregress_knn(xt,yt,k,nbootstraps,dt)

% NPREGRESS_KNN Performs nonparametric k-nearest-neighbors regression
%   [x,f,pval] = NPregress_knn(xt,yt,binedges,nbootstraps,dt) performs 
% nonparametric regression of 'yt' onto 'xt' using 'k' nearest neighboring 
% elements in 'xt' at a time.
%
% 'nbootstraps' specifies the number of bootstrap repetitions used to 
% compute standard error of the mean of the estimator.
if nargin<4, nbootstraps = []; dt = 1;
elseif nargin<5, dt = 1; end
if isempty(dt), dt = 1; end
if ~isempty(nbootstraps), compute_sem = 1; end

[xt,indx] = sort(xt);
yt = yt(indx);
n = length(xt);
nbins = length(k:k:n-k); % binning informed by data (resolution = k) --> finer resolution => slower

%% determine tuning function
if ~compute_sem % just return the means
    xval = xt(k:k:n-k); % estimate f(x) at every kth observation in x
    fval = zeros(nbins,1);
    for i=1:length(xval)
        xi = xt(i*k - (k-1):i*k + k); % sufficient to search +/-k elements around xval(i)
        yi = yt(i*k - (k-1):i*k + k);
        fval(i) = sum(yi(knnsearch(xi,xval(i),'k',k)))/(k*dt); % k-nearest
    end
    x.mu = xval;
    f.mu = fval;
else % obtain both mean and sem by bootstrapping (slow)
    x_mu = zeros(nbootstraps,nbins);
    f_mu = zeros(nbootstraps,nbins);
    for j=1:nbootstraps
        sampindx = sort(randsample(1:n,n,true));  % sample with replacement
        xt_samp = xt(sampindx); yt_samp = yt(sampindx);
        x_mu(j,:) = xt_samp(k:k:n-k); % estimate f(x) at every kth observation in x
        for i=1:length(x_mu(j,:))
            xi = xt_samp(i*k - (k-1):i*k + k); % sufficient to search +/-k elements around xval(i)
            yi = yt_samp(i*k - (k-1):i*k + k);
            f_mu(j,i) = sum(yi(knnsearch(xi,x_mu(j,i),'k',k)))/(k*dt); % k-nearest
        end
    end
    x.mu = mean(x_mu);
    x.sem = std(x_mu);
    f.mu = mean(f_mu); % mean
    f.sem = std(f_mu); % standard error of the mean
end