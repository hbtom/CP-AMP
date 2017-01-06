% This function sets the EM options to defaults.
function EMopt = CP_EMOpt()

%We limit the number of iterations after the EM update to reduce run time
EMopt.lim_nit = 100; %limit iterations

%Define default values
%Toggle 'heavy_tailed mode'
EMopt.heavy_tailed = false;

%Set default SNR (in dB) used in the initialization of the noise variance
EMopt.SNRdB = 10;

%Initial values (note that the mean of the entries of X_I is initialized
%with zero)
EMopt.noise_var = []; %initial noise variance of X_I

%Options to control learning
%Toggle learning of mean, variances, weights, sparsity rate, and noise variance
EMopt.learn_noisevar = true; %learn the variance of the AWGN
EMopt.learn_mean = true; %learn the mean of the X_I entries
EMopt.learn_var = true; %learn the variance of the X_I entries
EMopt.learn_weights = true; %learn the weights for GM prior only
EMopt.learn_lambda = true; %learn the the sparsity rate for sparse prior

EMopt.sig_dim = 'row'; %learn a single variances for Xi entries 
%(joint) or a different variance per (row) or (column: col)

%Set default number of mixture components
EMopt.L = 3;

%Set minium allowed variance of a GM component
EMopt.minVar = 1e-5;

%Set maximum number of iterations for learning model order
EMopt.maxLiter = 3;

%Iteration control. Note that the difference between maxEMiter and
%maxEMiterInner is relevant for the rank selection.
EMopt.maxEMiter = 20; %maximum number of EM cycles
EMopt.maxEMiterInner = 1; %maximum number of EM iterations for a given rank estimate
EMopt.tmax = 20; %first EM iteration to use full expression for noise variance update
%typically works well to set this equal to maxEMiter

%EMopt.EMtol = opt.tol; %convergence tolerance
EMopt.maxTol = 1e-5; %largest allowed tolerance for a single EM iteration

%Tolerances
EMopt.EMtol = min(10.^(-EMopt.SNRdB/10-1),EMopt.maxTol);

%Rank learning mode
%Set to 0 to disable rank learning
%Set to 1 to use the AICc rank learning
EMopt.learnRank = 1;

%By default, the maximum rank is set to the maximum rank for which the
%matrix can be uniquely determined based on the number of provided
%measurements
EMopt.rankMax = 30; %maximum allowed rank

%Optionts for rank selection (no effect when EMopt.learnRank == 0)
EMopt.rankStep = 1; %amount to increase rank
EMopt.rankStart = 1; %initial rank for rank learning

return
