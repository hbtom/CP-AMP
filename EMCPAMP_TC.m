function [estFin,optFin,EMoptFin,estHist] = EMCPAMP_TC(Y, problem, opt)

%EMCPAMP_TC:  Expectation-Maximization CP-AMP for tatrix completion.
%
%EM-CP-AMP tunes the parameters of the distributions on X^(i), and Y|Z
%assumed by CP-AMP using the EM algorithm. This version of the code
%assumes that the entries of X^(1) are N(0,1), while the entries of X^(i>1)
%are Gaussian with variance (and possibly mean) to be learned. The noise is
%assumed to be AWGN with unknown variance. Optionally, a procedure based on
%AICc may be used to learn the underlying CP-rank. This function is an
%example. Similar procedures with different choices for the priors can be
%created with minimal changes to the code.
%
% INPUTS:
% -------
% Y: the noisy tensor matrix. May be passed as a full tensor, or a sparse 
%   tensor as in sparse(Y), or as a tensor containing the observed entries. 
%   The locations of these entries are defined in the rowLocations and
%   columnLocations fields of the problem object.
% problem: An objet that specifies the problem setup, including the tensor
%   dimensions and observation locations
% opt (optional):  A set of options of the class CPAMPOpt.
% EMopt (optional): A structure containing several fields. If the input is
%   omitted or empty, the values in defaultOptions defined below will be
%   used. The user may pass a structure containing any or all of these 
%   fields to replace the defaults. Fields not included in the user input 
%   will be replaced with the defaults
%
% OUTPUTS:
% --------
% estFin: Structure containing final CP-AMP outputs
% optFin: The CPAMPOpt object used
% EMoptFin: The EM options used
% estHist: Structure containing per iteration metrics about the run


%% Handle Options

%Get problem dimensions from problem object
DIM = problem.DIM;
I = length(DIM);
R = problem.R;

%Indices of observed entries
if ~isempty(problem.rowLocations)
    omega = sub2ind(DIM,problem.rowLocations,problem.columnLocations);
else
    warning('Sampling locations not specified- assuming all entries are observed.') %#ok<WNTAG>
    omega = 1:prod(DIM);
end
Ns = numel(omega); %number of measurements

%Create options if not provided
if (nargin < 3)
    opt = [];
end

%Handle empty options object
if (isempty(opt))
    opt = CPAMPOpt();
end

%Set default EM options
EMopt = CP_EMOpt();

%% Initial Setup

%Rank learning mode
%Set to 0 to disable rank learning
%Set to 1 to use the AICc rank learning method. This approach tends to be
%very robust, particularly when the singular values of the true matrix tail
%off smoothly.
%If R is provided, by default we do not try to learn the rank
if ~isempty(R)
    EMopt.learnRank = 0; 
else
    EMopt.learnRank = 1;
end

%Set initial rank R when enable rank learning
if EMopt.learnRank == 1
    R = EMopt.rankStart;
    problem.R = R;
end

%No need to learn sparse rate when deal with non-sparse prior
EMopt.learn_lambda = false;

%Compute noise variance if not provided
if ~isfield(EMopt,'noise_var') || isempty(EMopt.noise_var)
    EMopt.noise_var = sum(abs(Y(:)).^2)/(numel(omega)*101);
end

%History
histFlag = false;
if nargout >=4
    histFlag = true;
    
    estHist.errZ = [];
    estHist.val = [];
    estHist.step = [];
    estHist.pass = [];
    estHist.timing = [];
end;

%Set initial noise variance
nuw = EMopt.noise_var;

%Set initial mean of X_I to 0
meanX = zeros(DIM(I),R);

%Set initial sparse level (lambda) of X_I to 1
lambda = ones(DIM(I),R);

%Compute the variance of X_I
nuX = (norm(Y(:),'fro')^2/numel(omega) - nuw)/R;

%Set iterative parameter 
if ~isfield(EMopt,'tmax') || isempty(EMopt.tmax)
    tmax = EMopt.maxEMiter;
else
    tmax = EMopt.tmax;
end

%Initialize loop
t = 0;
tInner = 0;
stop = 0;

%Placeholder initializations
Xhat = cell(1,I);
Xvar = cell(1,I);

%Initialize Xhat
for i = 1:I-1
    Xhat{i} = randn(DIM(i),R);
    opt.Xvar0{i} = ones(DIM(i),R);
end

%Xhat{I} = sqrt(nuX{I})*randn(DIM(I),R);
Xhat{I} = zeros(DIM(I),R); %seems to perform better

opt.Xhat0 = Xhat;
opt.Xvar0{I} = nuX*ones(DIM(I),R);

%Set initial step size small
opt.step = opt.stepMin;

%Original tolerance
tol0 = opt.tol;

%Ensure that EM outputs are calculated
opt.saveEM = 1;

%Outer loop for rank learning
bestVal = -inf;
rankStop = false;
zhatLast = 0;
SNR = 100;
outerCounter = 0;
zhatOld = 0;

%% Main loop

while ~rankStop
    
    % A structure containing all the values needed to warm start
    state = [];
    
    outerCounter = outerCounter + 1;
    
    %EM iterations
    %The < 2 condition is so that we can do a final iteration using the full
    %noise variance update without repeating code
    while stop < 2
        
        %Start timing
        tstart = tic;
        
        %Estimate SNR
        SNRold = SNR;
        
        %Compute SNR
        if t > 0
            opt.nit = EMopt.lim_nit; %limit the number of iterations after EM learning
            SNR = norm(zhat(omega),'fro')^2/norm(Y(omega) - zhat(omega),'fro')^2;
        end
        
        %Set tolerance for this iteration
        %tolNew = 1 / SNR;
        tolNew = min(max(tol0,1/SNR),EMopt.maxTol);
        opt.tol = tolNew;
        
        %Increment time exit loop if exceeds maximum time
        t = t + 1;
        tInner = tInner + 1;
        if tInner >= EMopt.maxEMiterInner || stop > 0
            stop = stop + 1;
        end
        
        %Prior on X_i (i<I)
        gX = cell(1,I);
        for i = 1:I-1
            gX{i} = AwgnEstimIn(0, 1);
        end
        
        %Prior on X_I
        gX{I} = AwgnEstimIn(meanX, nuX);
        
        %Log likelihood
        gOut = AwgnEstimOut(Y, nuw);
        
        %Stop timing
        t1 = toc(tstart);
        
        %Run CP-AMP
        [estFin2,~,estHist2, state] = CPAMP(gX, gOut, problem, opt, state);
        
        %Start timing
        tstart = tic;
        
        %Correct cost function
        estHist2.val = estHist2.val - 0.5*numel(omega)*log(2*pi*nuw);
        
        %Report progress
        if histFlag
            error_value = estHist2.errZ(end);
        else
            error_value = nan;
        end
        disp(['It ' num2str(t,'%04d')...
            ' nuX = ' num2str(mean(nuX(:)),'%5.3e')...
            ' meanX = ' num2str(mean(meanX(:)),'%5.3e')...
            ' tol = ' num2str(opt.tol,'%5.3e')...
            ' nuw = ' num2str(nuw,'%5.3e')...
            ' SNR = ' num2str(10*log10(SNR),'%03.2f')...
            ' Error = ' num2str(error_value,'%05.4f')...
            ' numIt = ' num2str(length(estHist2.errZ),'%04d')])

        %Compute zhat
        zhat = double(ktensor(estFin2.Xhat));
        
        %Calculate the change in signal estimates
        norm_change = norm(zhat(:)-zhatOld(:))^2/norm(zhat(:))^2;
        zhatOld = zhat;
        
        %Check for estimate tolerance threshold
        if (norm_change < max(tolNew/10,EMopt.EMtol)) &&...
                ( (norm_change < EMopt.EMtol) ||...
                (abs(10*log10(SNRold) - 10*log10(SNR)) < 1))
            stop = stop + 1;
        end
        
        
        %Learn noise variance
        if EMopt.learn_noisevar
            
            %First, just the term based on the residual
            nuw = norm(Y(omega) - zhat(omega),'fro')^2/numel(omega);
            
            %Then the component based on zvar
            if tInner >= tmax || stop > 0
                if isscalar(estFin2.zvar)
                    nuw = nuw + estFin2.zvar;
                else
                    nuw = nuw + sum(estFin2.zvar(omega))/numel(omega);
                end
            end
        end
        
        %Estimate new X parameters
        [lambda, meanX, nuX] =...
            BG_update(estFin2.qhat{I}, estFin2.qvar{I},...
            lambda, meanX, nuX, EMopt);
        
        %Reinitialize CP-AMP estimator
        for i = 1: I
            opt.Xhat0{i} = estFin2.Xhat{i};
            opt.Xvar0{i} = estFin2.Xvar{i};
        end
        opt.shat0 = estFin2.shatOpt;
        opt.step = opt.stepMin;
        
        %Stop timing
        t2 = toc(tstart);
        
        %Output Histories if necessary
        if histFlag
            estHist.errZ = [estHist.errZ; estHist2.errZ];
            estHist.val = [estHist.val; estHist2.val];
            estHist.step = [estHist.step; estHist2.step];
            estHist.pass = [estHist.pass; estHist2.pass];
            if t == 1
                estHist.timing = [estHist.timing; t1 + t2 + estHist2.timing];
            else
                estHist.timing = [estHist.timing; t1 + t2 + estHist.timing(end) ...
                    + estHist2.timing];
            end
        end
        
    end
    
    %Start timing
    tstart = tic;
    
    %Save several quantities for this iteration
    rank_hist(outerCounter) = R; %#ok<AGROW>
    cost_hist(outerCounter) = estHist.val(end); %#ok<AGROW>
    error_hist(outerCounter) = estHist.errZ(end);%#ok<AGROW>
    
    %Compute the residual
    residualHist(outerCounter) = norm(Y(omega) - zhat(omega),'fro')^2; %#ok<AGROW>
    
    %Compute number of free parameters
    Ne = R*(sum(DIM) - R)-2*(I-1)+1;
    
    %Compute the various penalty functions
    AICc(outerCounter) = -Ns*log(residualHist(outerCounter)/Ns) - 2*Ns./(Ns - Ne - 1).*Ne; %#ok<AGROW>
    
    
    %Stop rank inflation if we are doing worse
    if AICc(outerCounter) < bestVal
        rankStop = true;
        disp(['Terminating, AICc was ' num2str(AICc(outerCounter),'%5.3e') ...
            ', estimated rank was ' num2str(rank_hist(outerCounter - 1))])
    else %otherwise, update estimate
        estFin.Xhat = estFin2.Xhat;
        estFin.Xvar = estFin2.Xvar;
        %Save final options
        optFin = opt;
        EMoptFin = EMopt;
    end
    bestVal = AICc(outerCounter);
    
    %Check max rank
    if R >= EMopt.rankMax
        rankStop = true;
    end
    
    %Reset stop
    stop = 0;
    tInner = 0;
    
    %Calculate the change in signal estimates
    norm_change = norm(zhat(:)-zhatLast(:),'fro')^2/norm(zhat(:),'fro')^2;
    if norm_change < EMopt.EMtol
        rankStop = true;
    end
    zhatLast = zhat;
    
    %Check total EM iterations
    if t >= EMopt.maxEMiter
        rankStop = true;
    end
    
    %Check to see if rank learning is enabled
    if EMopt.learnRank ~= 1
        rankStop = true;
    end
    
    %Stop if we are over-parameterized
    if Ne > numel(omega)
        rankStop = true;
    end
    
    %If we are not stopping, increase the rank
    if ~rankStop
        
        %Increase by rank step
        R = R + EMopt.rankStep;
        problem.R = R;
        
        disp(['Increasing rank to ' num2str(R) ...
            ' AICc was ' num2str(AICc(outerCounter),'%5.3e')])
        
        %Expand
        for i = 1:I
%             if I > 3
                Xhat{i} = randn(DIM(i),R);
                Xvar{i} = ones(DIM(i),R);
%             else
%                 Xhat{i} = [estFin2.Xhat{i} sqrt(var(estFin2.Xhat{i}(:,end)))*randn(DIM(i),EMopt.rankStep)];
%                 Xvar{i} = [estFin2.Xvar{i} mean(estFin2.Xvar{i}(:))*ones(DIM(i),EMopt.rankStep)];
%             end
            opt.Xhat0{i} = Xhat{i};
            opt.Xvar0{i} = Xvar{i};
        end

%         Xhat{I} = [estFin2.Xhat{I} zeros(DIM(I),EMopt.rankStep)];
%         opt.Xhat0{I} = Xhat{I};
        
        %Expand nuX
        nuX = [nuX mean(nuX(:,end))*ones(DIM(I),EMopt.rankStep)]; %#ok<AGROW>
        meanX = [meanX mean(meanX(:,end))*ones(DIM(I),EMopt.rankStep)]; %#ok<AGROW>
        
        %Fix lambda
        lambda = ones(DIM(I),R);
    end
    
    %Stop timing
    t3 = toc(tstart);
    
    %Add the time in
    estHist.timing(end) = estHist.timing(end) + t3;
    
end


%% Cleanup

%Save the history
estHist.AICc = AICc;
estHist.cost_hist = cost_hist;
estHist.residualHist = residualHist;
estHist.rank_hist = rank_hist;
estHist.error_hist = error_hist;



