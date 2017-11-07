function [x,f] = NPregress_knn(xt,yt,dt,k,nbins,nbootstraps)

% NPREGRESS_KNN Performs nonparametric k-nearest-neighbors regression
%   [x,f,pval] = NPregress_knn(xt,yt,binedges,nbootstraps,dt) performs 
% nonparametric regression of 'yt' onto 'xt' using 'k' nearest neighboring 
% elements in 'xt' at a time.
%
% 'nbootstraps' specifies the number of bootstrap repetitions used to 
% compute standard error of the mean of the estimator.
if nargin<6, nbootstraps = []; end
if nargin<5, nbins = []; end
if nargin<4, k = []; end
if nargin<3, dt = []; end

if isempty(dt), dt = 1; end
n = length(xt);
if isempty(k), k = round(sqrt(n)); end % k=sqrt(n) where n is the total no. of observations
if isempty(nbins), nbins = length(k:k:n-k); end % binning informed by data (resolution = k) --> finer resolution => slower
if ~isempty(nbootstraps), compute_sem = 1; end
p_binedges = linspace(0,100,nbins+1); % divide datapoints into 'nbins' number of quantiles
p_bincntrs = 0.5*(p_binedges(1:end-1) + p_binedges(2:end)); % percentile scores of individual quantiles

%% sort data for efficient search
[xt,indx] = sort(xt); yt = yt(indx);

%% determine tuning function
if ~compute_sem % just return the means
    xval = prctile(xt(k:n-k),p_bincntrs);
    fval = zeros(nbins,1);
    for i=1:length(xval)
        [~,nearestindx] = min(abs(xt-xval(i)));
        xi = xt(nearestindx - (k-1):nearestindx + k); % sufficient to search +/-k elements around bincentre
        yi = yt(nearestindx - (k-1):nearestindx + k);
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
        x_mu(j,:) = prctile(xt_samp(k:n-k),p_bincntrs);
        for i=1:length(x_mu(j,:))
            [~,nearestindx] = min(abs(xt_samp-x_mu(j,i)));
            xi = xt_samp(nearestindx - (k-1):nearestindx + k); % sufficient to search +/-k elements around bincentre
            yi = yt_samp(nearestindx - (k-1):nearestindx + k);
            f_mu(j,i) = sum(yi(knnsearch(xi,x_mu(j,i),'k',k)))/(k*dt); % k-nearest
        end
    end
    x.mu = mean(x_mu);
    x.sem = std(x_mu);
    f.mu = mean(f_mu); % mean
    f.sem = std(f_mu); % standard error of the mean
end