function [estFin, optFin, estHist, state] = ...
    CPAMP(gX, gOut, problem, opt, state) 
%% Setup
% Get options
if (nargin < 4)
    opt = CPAMPOpt();
elseif (isempty(opt))
    opt = CPAMPOpt();
end
nit     = opt.nit;              % number of iterations
nitMin  = opt.nitMin;           % minimum number of iterations
step    = opt.step;             % step size
stepMin = opt.stepMin;          % minimum step size
stepMax = opt.stepMax;          % maximum step size
stepFilter = opt.stepFilter;    % step filter setting, <1 for no effect
adaptStep = opt.adaptStep;      % adaptive step size
stepIncr = opt.stepIncr;        % step inc on succesful step
stepDecr = opt.stepDecr;        % step dec on failed step
stepWindow = opt.stepWindow;    % step size check window size
verbose = opt.verbose;          % Print results in each iteration
tol = opt.tol;                  % Convergence tolerance
stepTol = opt.stepTol;          % minimum allowed step size
pvarStep = opt.pvarStep;        % incldue step size in pvar/zvar
varNorm = opt.varNorm;          % normalize variances
compVal = adaptStep;            % only compute cost function for adaptive
maxBadSteps = opt.maxBadSteps;  % maximum number of allowed bad steps
maxStepDecr = opt.maxStepDecr;  % amount to decrease maxStep after failures
zvarToPvarMax = opt.zvarToPvarMax;  % maximum zvar/pvar ratio

%Determine requested outputs
saveEM = opt.saveEM;
saveHist = (nargout >= 3);
saveState = (nargout >= 4);

%Check for provided state
if nargin < 5
    state = [];
end

%Get problem dimensions
DIM = problem.DIM;
I = length(DIM);
R = problem.R;

%Check for partial observation of Z
rLoc = problem.rowLocations;
cLoc = problem.columnLocations;
maskFlag = ~isempty(rLoc);

%Check lengths
if length(rLoc) ~= length(cLoc)
    error('rowLocations and columnLocations must be same length')
end

%Setup for masked case
if maskFlag
    %Indices of observed entries
    omega = sub2ind(DIM,rLoc,cLoc);
    
    %Create mask matrix if not in sparse mode
    maskMatrix = zeros(DIM);
    maskMatrix(omega) = 1;
end

%Preallocate storage for estHist if user requires it
if (saveHist)
    estHist.errZ = zeros(nit,1);
    estHist.val = zeros(nit,1);
    estHist.step = zeros(nit,1);
    estHist.pass = false(nit,1);
    estHist.timing = zeros(nit,1);
end

%Assign Xivar mins
XivarMin = opt.XivarMin;

%Assign pvar mins
pvarMin = opt.pvarMin;

%% Initialization

%Placeholder initializations
Xhat = cell(1,I);
Xvar = cell(1,I);
XhatOpt = cell(1,I);
XvarOpt = cell(1,I);
XhatBarOpt = cell(1,I);
XhatBar = cell(1,I);
XhatBar(:) = {'0'};
Xhat2 = cell(1,I);
XhatBar2 = cell(1,I);
XhatBar2var = cell(1,I);
Xhat2var = cell(1,I);
Xhat2var1 = cell(1,2*I);
a = cell(1,I);
a2 = cell(1,I);
qhat = cell(1,I);
qvar = cell(1,I);
qhatFinal = cell(1,I);
qvarFinal = cell(1,I);
va = cell(1,I);
qGain = cell(1,I);
valInX = cell(1,I);

if isempty(state) %if no state is provided
    
    for i = 1:I
        %Initialize Xihat and Xivar
        [Xhat{i},Xvar{i}] = gX{i}.estimInit();
        
        %Handle case of scalar input distribution estimator on Xhat
        if (length(Xhat{i}) == 1) && i ~= I
            Xhat{i} = gX{i}.genRand([DIM(i) R]);
        end
        
        %Handle uniform variances
        if (length(Xvar{i}) == 1)
            Xvar{i} = repmat(Xvar{i},DIM(i),R);
        end
    end
    
    Xhat{I} = zeros(DIM(I),R); %seems to perform better
    
    %Replace these defaults with the warm start values if provided in the
    %options object (for EM learning)
    if ~isempty(opt.Xhat0)
        Xhat = opt.Xhat0;
        Xvar = opt.Xvar0;
        
        %Handle uniform variances
        if (length(Xvar{i}) == 1)
            Xvar{i} = repmat(Xvar{i},DIM(i),R);
        end
    end
    
    %Initialize valIn
    valIn = -inf;
    
    %Initialize valOpt empty
    valOpt = [];
    
    %Set pvar_mean to unity initially
    pvar_mean = 1;
    
    %Placeholder initializations
    shat = 0;
    svar = 0;
    pvarOpt = 0;
    zvarOpt = 0;
    
    %Address warm starting of shat0
    if ~isempty(opt.shat0)
        shat = opt.shat0;
    end
    
else %Use the provided state information
    
    %X variables
    Xhat = state.Xhat;
    Xvar = state.Xvar;
    XhatBar = state.XhatBar;
    XhatBarOpt = state.XhatBarOpt;
    XhatOpt = state.XhatOpt;
    
    %Old cost values
    valOpt = state.valOpt;
    
    %Cost stuff
    valIn = state.valIn;
    
    %Step
    step = state.step;
    
    %S variables
    shat = state.shat;
    svar = state.svar;
    shatOpt = state.shatOpt;
    shatNew = state.shatNew;
    svarOpt = state.svarOpt;
    svarNew = state.svarNew;
    
    %other variables
    pvar_mean = state.pvar_mean;
    pvarOpt = state.pvarOpt;
    zvarOpt = state.zvarOpt;
end

%Cost init
val = zeros(nit,1);
zhatOpt = 0;
testVal = inf;

%% Iterations

%Start timing first iteration
tstart = tic;

%Control variable to end the iterations
stop = false;
it = 0;
failCount = 0;

%Handle first step
if isempty(state)
    step1 = 1;
else
    step1 = step;
end

% Main iteration loop
while ~stop
    
    % Iteration count
    it = it + 1;
    
    % Check for final iteration
    if it >= nit
        stop = true;
    end
    
    %Precompute squares quantities for use, these change on every iteration
    for i = 1:I
        Xhat2{i} = abs(Xhat{i}).^2;
        Xhat2var{i} = Xhat2{i} + Xvar{i};
        Xhat2var1{2*i-1} = Xhat2{i};
        Xhat2var1{2*i} = Xvar{i};
    end
        
    zvar = 0;
    for i = 1:I
        Xhat2var1_temp = Xhat2var1;
        Xhat2var1_temp([2*i, setdiff(1:I,i)*2-1]) = [];
        zvar = zvar + double(ktensor(Xhat2var1_temp));%Compute zvar
    end
    %Compute pvar
    pvar = double(ktensor(Xhat2var) - ktensor(Xhat2));

    %Include pvar step
    if pvarStep
        pvar = step1*pvar + (1-step1)*pvarOpt;
        zvar = step1*zvar + (1-step1)*zvarOpt;
    end
    
    %Update zhat
    zhat = double(ktensor(Xhat));
    
    % Continued output step
    phat = zhat - shat.*(zvar/pvar_mean);
        
    % Compute log likelihood at the output and add it the total negative
    % K-L distance at the input.
    if (compVal)
        valOut_temp = maskMatrix .* gOut.logLike(zhat,pvar);
        valOut = sum(valOut_temp(:));
        val(it) = valOut + valIn;
    end
    
    % Determine if candidate passed
    if ~isempty(valOpt)
        %Check against worst value in last stepWindow good steps
        stopInd = length(valOpt);
        startInd = max(1,stopInd - stepWindow);
        
        %Check the step
        pass = (val(it) > min(valOpt(startInd:stopInd))) ||...
            ~adaptStep || (step <= stepMin);
    else
        pass = true;
    end
    
    %Save the step size and pass result if history requested
    if saveHist
        estHist.step(it) = step;
        estHist.pass(it) = pass;
    end
    
    % If pass, set the optimal values and compute a new target shat and
    % snew.
    if (pass)
        
        %Slightly inrease step size after pass if using adaptive steps
        if adaptStep
            step = stepIncr*step;
        end
        
        % Set new optimal values
        shatOpt = shat;
        svarOpt = svar;
        pvarOpt = pvar;
        zvarOpt = zvar;
        
        XhatBarOpt = XhatBar;
        XhatOpt = Xhat;
        
        %Bound pvar
        pvar = max(pvar, pvarMin);
        
        %We keep a record of only the succesful step valOpt values
        valOpt = [valOpt val(it)]; %#ok<AGROW>
        
        %Compute mean of pvar
        if varNorm
            pvar_mean = mean(pvar(:));
        end
        
        % Output nonlinear step
        [zhat0,zvar0] = gOut.estim(phat,pvar);
        
        %Compute 1/pvar
        pvarInv = pvar_mean ./ pvar;
        
        %Update the shat quantities
        pvarInvMask = pvarInv .* maskMatrix;
        shatNew = pvarInvMask.*(zhat0-phat);
        svarNew = pvarInvMask.*(1-min(zvar0./pvar,zvarToPvarMax));
        
        %Enforce step size bounds
        step = min([max([step stepMin]) stepMax]);
        
    else
        
        %Check on failure count
        failCount = failCount + 1;
        if failCount > maxBadSteps
            failCount = 0;
            stepMax = max(stepMin,maxStepDecr*stepMax);
        end
        % Decrease step size
        step = max(stepMin, stepDecr*step);
        
        %Check for minimum step size
        if step < stepTol
            stop = true;
        end
    end
    
    % Save results
    if (saveHist)
        
        %Record timing information
        if it > 1
            estHist.timing(it) = estHist.timing(it-1) + toc(tstart);
        else
            estHist.timing(it) = toc(tstart);
        end
        
        %Compute the Z error only if needed
        estHist.errZ(it) = opt.error_function(zhat);
        estHist.val(it) = val(it);
    end
    
    
    % Check for convergence if step was succesful
    if pass
        if any(isnan(zhat(:))) || any(isinf(zhat(:)))
            stop = true;
        else
            testVal = norm(zhat(:) - zhatOpt(:)) / norm(zhat(:));
            if (it > 1) && (testVal < tol)
                stop = true;
            end
        end
        
        %Set other optimal values- not actually used by iterations
        XvarOpt = Xvar;
        zhatOpt = zhat;
        
        %Save EM variables if requested
        if saveEM && it ~= 1
            qhatFinal{i} = qhat{i};
            qvarFinal{i} = pvar_mean*qvar{i};
            zvarFinal = zvar0;
            pvarFinal = pvar;
        end
        
    end
    
    % Print results
    if (verbose)
        if ~saveHist
            fprintf(1,'it=%3d value=%12.4e step=%f\n', it, testVal, step1);
        else
            fprintf(1,...
                'it=%3d value=%12.4e errZ=%f step=%f\n', it, testVal,...
                estHist.errZ(it), step1);
        end
    end
    
    %Start timing next iteration
    tstart = tic;
    
    % Create new candidate shat
    if it > 1 || ~isempty(state)
        step1 = step;
        if stepFilter >= 1
            step1 = step1*it/(it+stepFilter);
        end
    end
    shat = (1-step1)*shatOpt + step1*shatNew;
    svar = (1-step1)*svarOpt + step1*svarNew;
    
    for i = 1:I
        XhatBar{i} = (1-step1)*XhatBarOpt{i} + step1*XhatOpt{i};
        XhatBar2{i} = abs(XhatBar{i}).^2;
        XhatBar2var{i} = abs(XhatBar{i}).^2+Xvar{i};
    end
    
%==========================================================================Input linear step
    for i = 1:I
        XhatBar2_temp = XhatBar2;
        XhatBar2_temp(i) = [];
        a2{i} = yqtensor(XhatBar2_temp);
        
        qvar{i} = 1./double((ttt(tensor(svar),tensor(a2{i}),setdiff(1:I,i),1:I-1)));
        qvar{i}(qvar{i} > opt.varThresh) = opt.varThresh;
        
        Xhat2var_temp = XhatBar2var;
        Xhat2var_temp(i) = []; 
        va{i} = yqtensor(Xhat2var_temp)-a2{i};
        
        qGain{i} = (1 - (qvar{i}.*double((ttt(tensor(svar),tensor(va{i}),setdiff(1:I,i),1:I-1)))));
        qGain{i} = min(1,max(0,qGain{i}));
        
        XhatBar_temp = XhatBar;
        XhatBar_temp(i) =[];
        a{i} = yqtensor(XhatBar_temp);
        
        qhat{i} = XhatBar{i}.*qGain{i} + qvar{i}.*double((ttt(tensor(shat),tensor(a{i}),setdiff(1:I,i),1:I-1)));
        
        qvar{i} = max(qvar{i},XivarMin);
        
        valIn = 0;
        if compVal
            [Xhat{i},Xvar{i},valInX{i}] = gX{i}.estim(qhat{i}, qvar{i}*pvar_mean);
            %Update valIn
            valIn = valIn + sum(valInX{i}(:));
        else
            [Xhat{i},Xvar{i}] = gX{i}.estim(qhat{i}, qvar{i}*pvar_mean);
        end
    end

%==========================================================================Input nonlinear step

    %Don't stop before minimum iteration count
    if it < nitMin
        stop = false;
    end
    
end

%% Save the final values

%Save the options object that was used
optFin = opt;

%Estimates the matrix factors
estFin.Xhat = XhatOpt;
estFin.Xvar = XvarOpt;

if saveEM
    estFin.qhat = qhatFinal;
    estFin.qvar = qvarFinal;
    estFin.zvar = zvarFinal;
    estFin.phat = phat;
    estFin.pvar = pvarFinal;
    estFin.shatOpt = shatOpt;
end

%% Cleanup estHist

%Trim the outputs if early termination occurred
if saveHist && (it < nit)
    estHist.errZ = estHist.errZ(1:it);
    estHist.val = estHist.val(1:it);
    estHist.step = estHist.step(1:it);
    estHist.pass = estHist.pass(1:it);
    estHist.timing = estHist.timing(1:it);
end

%% Save the state

if saveState
    %X variables
    state.Xhat = Xhat;
    state.Xvar = Xvar;
    state.XhatBar = XhatBar;
    state.XhatBarOpt = XhatBarOpt;
    state.XhatOpt = XhatOpt;
    
    %Old cost values
    state.valOpt = valOpt;
    
    %Cost stuff
    state.valIn = valIn;
    
    %Step
    state.step = step;
    
    %S variables
    state.shat = shat;
    state.svar = svar;
    state.shatOpt = shatOpt;
    state.svarOpt = svarOpt;
    state.shatNew = shatNew;
    state.svarNew = svarNew;
    
    %other variables
    state.pvarOpt = pvarOpt;
    state.pvar_mean = pvar_mean;
    state.zvarOpt = zvarOpt;
end


