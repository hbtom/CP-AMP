%Yangqing Li
%27 Nov 2016
%This is an example to illustrate Tensor Completion using CP-AMP.

%% Clean Slate
clear all
close all
clc
randn('state',0); rand('state',0); %#ok<RAND>

%This code requires the Tensor Toolbox for Matlab, which can be downloaded 
%from http://www.sandia.gov/~tgkolda/TensorToolbox/ (tested using V2.5). 
addpath(genpath('tensor_toolbox_2.5/.'));

%It also requires the GAMPMATLAB package for Matlab, which can be downloaded 
%from: https://sourceforge.net/projects/gampmatlab/ (tested using 20161005).
addpath(genpath('gampmatlab20161005/.'));


%% Generate the Unknown Low-CP-Rank Tensor

%Generate raw data. The Low-CP-rank R tensor will be of size DIM 
DIM = [50,50,50];

R = 3;

%Generate the factor matrices
X = cell(1,length(DIM));
for m=1:length(DIM)
	X{m} =  gaussSample(ones(R,1), eye(R), DIM(m));
end

lambda = ones(1,R);
% lambda = rand(1,R);

%Noise free signal tensor from factor matrices
Z = double(ktensor(lambda',X));

%Define the error function for computing normalized mean square error.
%CP-AMP will use this function to compute NMSE for each iteration
error_function = @(qval) 20*log10(norm(qval(:) - Z(:)) / norm(Z(:)));

%% AWGN Noise

%We will corrupt the observations of Z, denoted Y, with AWGN. We will
%choose the noise variance to achieve an SNR (in dB) of
SNR = 50;

%Determine the noise variance that is consistent with this SNR
nuw = var(Z(:))*10^(-SNR/10);
% nuw1 = norm(reshape(Z,[],1))^2/prod(DIM)*10^(-SNR/10);% Same as above

%Generate noisy data
Y = Z + sqrt(nuw)*randn(size(Z));

%% Observe a fraction of the noisy matrix entries

%For tensor completion, we observe only a fraction of the entries of Y. We
%denote this fraction as p1
p1 = 0.10;

%Choose a fraction p1 of the entries of Y to keep. Omega has the same 
%dimension as Y, and is a tensor of logicals that will store the keeped
%locations
omega = false(size(Z));
ind = randperm(prod(DIM));
omega(ind(1:ceil(p1*prod(DIM)))) = true;

%Set the unobserved entries of Y to zero
Y(~omega) = 0;

%% Define options for CP-AMP

%Set options
opt = CPAMPOpt; %initialize the options object with defaults

%Provide CP-AMP the error function for plotting NMSE
opt.error_function = error_function;

%Specify the problem setup for CP-AMP, including the tensor dimensions and
%sampling locations. Notice that the CP-rank R can be learned by the EM 
%code and does not need to be provided in that case. We set it here for the 
%the use of the simplified codes which assume a known rank
problem.DIM = DIM;
problem.R = R;
[problem.rowLocations,problem.columnLocations] = find(omega);

%Initialize results as empty. This struct will store the results for each
%algorithm
results = [];


%% Run CP-AMP

%First, we will run CP-AMP with knowledge of the true distributions. To do
%this, we create objects that represent the priors and log likelihood.

%Prior distribution on factor matrix Xi is Gaussian
for r = 1:length(DIM)
    eval(['gX{',num2str(r),'}=AwgnEstimIn(0, 1);']);
end
%Log likelihood is Gaussian, i.e. we are considering AWGN
gOut = AwgnEstimOut(Y, nuw);

%Set the number of iterations
opt.nit = 1000;

%Run CPAMP
disp('Starting CP-AMP')
tstart = tic;
[~,~,estHist] = CPAMP(gX, gOut, problem, opt);
tCPAMP = toc(tstart);

%Save results for ease of plotting
loc = length(results) + 1;
results{loc}.name = 'CP-AMP';
results{loc}.err = estHist.errZ(end);
results{loc}.time = tCPAMP;
results{loc}.errHist = estHist.errZ;
results{loc}.timeHist = estHist.timing;

%% Run EM-CP-AMP

%Now we run EM-CP-AMP. Notice that EM-CP-AMP requires only the
%observations Y, the problem object, and (optional) options. The model 
%parameters are not provided. EM-CP-AMP learns these parameters, e.g.,
%the noise level.

%Set the number of iterations that CP-AMP is allowed during the first EM 
%iteration to reduce run time
opt.nit = 300; %limit iterations

disp('Starting EM-CP-AMP')
disp(['The true value of nuw was ' num2str(nuw)])
%Run EM-CP-AMP
tstart = tic;
[estFin,~,~,estHist] = EMCPAMP_TC(Y,problem,opt);
tEMCPAMP = toc(tstart);

%Save results for ease of plotting
loc = length(results) + 1;
results{loc}.name = 'EM-CP-AMP'; %#ok<*AGROW>
results{loc}.err = opt.error_function(double(ktensor(estFin.Xhat)));
results{loc}.time = tEMCPAMP;
results{loc}.errHist = estHist.errZ;
results{loc}.timeHist = estHist.timing;

%% Run EM-CP-AMP with rank learning using penalized log-likelihood maximization

%Rank learning strategy: starts with a small rank and gradually increases
%the rank. At each tested rank, we evalute the AICc criteria.
%When this criteria stops improving, we take the rank corresponding to the
%best value of AICc as our rank estimate.

%This approach is tuned fairly conservatively to ensure good performance.
%It works well on noisy data and problems where the singular values tail
%off slowly without a clear-cut rank. 

%Since the default does not specify rank R in the problem setup, so we can
%simply set
problem.R = [];

%Set the number of iterations that CP-AMP is allowed during the first EM 
%iteration to reduce run time
opt.nit = 100; %limit iterations

disp('Starting EM-CP-AMP with rank learning using penalized log-likelihood maximization')
disp(['Note that the true rank was ' num2str(R)])
%Run EM-CP-AMP with rank learning
tstart = tic;
[estFin,~,~,estHist] = EMCPAMP_TC(Y,problem,opt);
tREMCPAMP = toc(tstart);

%Save results for ease of plotting
loc = length(results) + 1;
results{loc}.name = 'EM-CP-AMP (pen. log-like)'; %#ok<*AGROW>
results{loc}.err = opt.error_function(double(ktensor(estFin.Xhat)));
results{loc}.time = tREMCPAMP;
results{loc}.errHist = estHist.errZ;
results{loc}.timeHist = estHist.timing;

%% Show Results

%Display the contents of the results structure
results{:}  %#ok<NOPTS>








