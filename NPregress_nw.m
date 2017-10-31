function [x,f] = NPregress_nw(xt,yt,kernel,bandwidth,nbootstraps,dt)

% NPREGRESS_NW Performs nonparametric Nadaraya-Watson kernel regression
%   [x,f,pval] = NPregress_nw(xt,yt,kernel,nbootstraps,dt) performs 
% nonparametric regression of 'yt' onto 'xt', smoothed using the kernel 
% specified by 'kernel'.
%
% 'nbootstraps' specifies the number of bootstrap repetitions used to 
% compute standard error of the mean of the estimator.
if nargin<4, bandwidth = []; nbootstraps = []; dt = 1;
elseif nargin<5, nbootstraps = []; dt = 1;
elseif nargin<6, dt = 1;
end
if isempty(dt), dt = 1; end
if ~isempty(nbootstraps), compute_sem = 1; end
if isempty(bandwidth), bandwidth = range(xt)/50; end % kernel bandwidth is 2% of the total range

%% define kernel function
if strcmp(kernel,'Uniform'), kernel = @(x,mu,bandwidth) (1/2)*(abs((x - mu)/bandwidth) < 1);
elseif strcmp(kernel,'Epanechnikov'), kernel = @(x,mu,bandwidth) (3/4)*(1-((x - mu)/bandwidth).^2).*(abs((x - mu)/bandwidth) < 1);
elseif strcmp(kernel,'Biweight'), kernel = @(x,mu,bandwidth) (15/16)*((1-((x - mu)/bandwidth).^2).^2).*(abs((x - mu)/bandwidth) < 1);
elseif strcmp(kernel,'Gaussian'), kernel = @(x,mu,bandwidth) exp(-(((x - mu)/bandwidth).^2)/2);
end

n = length(xt);
nbins = round(range(xt)/bandwidth); % values of x at which to estimate f(x) -- 1 point per bandwidth

%% determine tuning function
if ~compute_sem % just return the means
    xval = linspace(min(xt),max(xt),nbins);
    kval = cell2mat(arrayfun(@(xi) kernel(xi,xt,bandwidth),xval,'UniformOutput',false));
    x.mu = xval;
    f.mu = sum(kval.*repmat(yt,[1 nbins]))./(sum(kval)*dt);
else % obtain both mean and sem by bootstrapping (slow)
    x_mu = zeros(nbootstraps,nbins);
    f_mu = zeros(nbootstraps,nbins);
    for j=1:nbootstraps
        sampindx = sort(randsample(1:n,n,true));  % sample with replacement
        xt_samp = xt(sampindx); yt_samp = yt(sampindx);
        xval = linspace(min(xt_samp),max(xt_samp),nbins);
        kval = cell2mat(arrayfun(@(xi) kernel(xi,xt_samp,bandwidth),xval,'UniformOutput',false));
        x_mu(j,:) = xval;
        f_mu(j,:) = sum(kval.*repmat(yt_samp,[1 nbins]))./(sum(kval)*dt);
    end
    x.mu = mean(x_mu);
    x.sem = std(x_mu);
    f.mu = mean(f_mu); % mean
    f.sem = std(f_mu); % standard error of the mean
end