classdef CPAMPOpt
    % Options for the CP-AMP optimizer.
    
    properties
        
        %***** General Options

        % Show progress
        verbose = true;%false true
        
        %Return additional outputs for EM learning when run EM-CP-AMP.
        %(no effect for CP-AMP with the knowledge of true distributions)
        saveEM = true;
        
        %Number of iterations
        nit = 150;
        
        %Minimum number of iterations
        nitMin = 0; %0 for no effect
        
        %Specify a convergence tolerance. Iterations are terminated when
        %the norm of the differnece in two iterates divided by the norm of
        %the current iterate falls below this threshold. 
        tol = 1e-6; %Set to -1 to disable
        
        %Error function. This is a function handle that acts on Zhat to
        %compute an NMSE error.
        error_function = @(q) inf;
        
        %Option to "normalize" variances for computation. May improve
        %robustness in some cases. Use not recommended.
        varNorm = false;
        
        %***** Step Size Control
        
        %Logical flag to include a step size in the pvar/zvar calculation.
        %This momentum term often improves numerical performance. On by
        %defualt.
        pvarStep = true;
        
        %Initial step size, or fixed size for non-adaptive steps
        step = 0.05;
        
        % Enable adaptive step size
        adaptStep = true;%false true
        
        % Minimum step size
        stepMin = 0.05;
        
        % Maximum step size
        stepMax = 0.5;
        
        % Multiplicative step size increase, when successful
        stepIncr = 1.1;
        
        % Multiplicative step size decrease, when unsuccessful
        stepDecr = 0.5;
        
        %Maximum number of allowed failed steps before we decrease stepMax,
        %inf for no effect
        maxBadSteps = inf;
        
        %Amount to decrease stepMax after maxBadSteps failed steps, 1 for
        %no effect
        maxStepDecr = 0.5;
        
        %Create a window for the adaptive step size test. Setting this to
        %zero causes it to have no effect. For postive integer values,
        %creats a moving window of this length when checking the step size
        %acceptance criteria. The new value is only required to be better
        %than the worst in this window, i.e. the step size criteria is not
        %required to monotonicaly increase.
        stepWindow = 1;
        
        % Iterations are termined when the step size becomes smaller
        % than this value. Set to -1 to disable
        stepTol = -1;
        
        %This is a filter value on the steps. It slows down the early steps
        %in a smooth way. Set to less than 1 for no effect, In particular,
        %the steps are computed as step1 = step*it/(it + stepFilter)
        stepFilter = 0;
        
        %Minimum variances. See code for details of use.
        pvarMin = 1e-13;
        XivarMin = 0;
        zvarToPvarMax = 0.99;% prevents negative svar, should be near 1
        
        %Variance threshold for rvar and qvar, set large for no effect
        varThresh = 1e6;
        
        %***** Initialization
        
        %Provide initial guesses for Xhat,Xvar,shat0. If these are set to
        %empty, then the appropriate method of the input estimator is
        %called to initialize these values. This functionality is useful
        %for warm starting the algorithm when not providing the full state.
        Xhat0 = [];
        Xvar0 = [];
        shat0 = [];
    end
    
    methods
        
        % Constructor with default options
        function opt = CPAMPOpt(varargin)
            if nargin == 0
                % No custom parameters values, thus create default object
                return
            elseif mod(nargin, 2) == 0
                % User is providing property/value pairs
                names = fieldnames(opt);    % Get names of class properties
                
                % Iterate through every property/value pair, assigning
                % user-specified values.  Note that the matching is NOT
                % case-sensitive
                for i = 1 : 2 : nargin - 1
                    if any(strcmpi(varargin{i}, names))
                        % User has specified a valid property
                        propName = names(strcmpi(varargin{i}, names));
                        opt.(propName{1}) = varargin{i+1};
                    else
                        % Unrecognized property name
                        error('CPAMPOpt: %s is an unrecognized option', ...
                            num2str(varargin{i}));
                    end
                end
                return
            else
                error(['The CPAMPOpt constructor requires arguments ' ...
                    'be provided in pairs, e.g., CPAMPOpt(''adaptStep'',' ...
                    ' false, ''nit'', 50)'])
            end
        end
    end
    
end
