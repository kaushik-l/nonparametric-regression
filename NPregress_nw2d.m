function [x,f] = NPregress_nw2d(xt,yt,dt,kernel,bandwidth,nbins)

% NPREGRESS_NW Performs nonparametric Nadaraya-Watson kernel regression
%   [x,f,pval] = NPregress_nw(xt,yt,kernel,nbootstraps,dt) performs 
% nonparametric regression of 'yt' onto 'xt', smoothed using the kernel 
% specified by 'kernel'.
%
% 'nbootstraps' specifies the number of bootstrap repetitions used to 
% compute standard error of the mean of the estimator.
if nargin<6, nbins = []; end
if nargin<5, bandwidth = []; end
if nargin<4, kernel = []; end
if nargin<3, dt = []; end

if isempty(dt), dt = 1; end
xt1 = xt(:,1); xt2 = xt(:,2);
n = length(xt1);
if isempty(kernel), kernel = 'Gaussian'; end
if isempty(bandwidth), bandwidth = [(2.5/100)*range(xt1) ; (2.5/100)*range(xt2)]; end
if isempty(nbins), nbins = [round(range(xt1)/bandwidth(1)); round(range(xt2)/bandwidth(2))]; end

%% define kernel function
if strcmp(kernel,'Uniform'), kernel = @(x,mu,bandwidth) (1/2)*(abs((x - mu)/bandwidth) < 1);
elseif strcmp(kernel,'Epanechnikov'), kernel = @(x,mu,bandwidth) (3/4)*(1-((x - mu)/bandwidth).^2).*(abs((x - mu)/bandwidth) < 1);
elseif strcmp(kernel,'Biweight'), kernel = @(x,mu,bandwidth) (15/16)*((1-((x - mu)/bandwidth).^2).^2).*(abs((x - mu)/bandwidth) < 1);
elseif strcmp(kernel,'Gaussian'), kernel = @(x,mu,bandwidth) exp(-(((x - mu)/bandwidth).^2)/2);
end

%% determine tuning function
binedges = linspace(min(xt),max(xt),nbins+1);
xval = 0.5*(binedges(1:end-1) + binedges(2:end));
kval = cell2mat(arrayfun(@(xi) kernel(xi,xt,bandwidth),xval,'UniformOutput',false));
x.mu = xval;
f.mu = sum(kval.*repmat(yt,[1 nbins]))./(sum(kval)*dt);