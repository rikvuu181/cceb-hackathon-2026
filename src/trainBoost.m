function mdl = trainBoost(X, y, cfg)
% Fit a gradient-boosted tree ensemble with the settings in cfg.
% cfg fields: LearnRate, MaxNumSplits, MinLeafSize, NumCycles, LogTarget.
    % Each split considers 70% of the predictors, which decorrelates the trees a little
    t = templateTree("MaxNumSplits", cfg.MaxNumSplits, "MinLeafSize", cfg.MinLeafSize, ...
        "NumVariablesToSample", round(0.7 * width(X)));
    if cfg.LogTarget
        y = log(y);
    end
    mdl = fitrensemble(X, y, "Method", "LSBoost", "NumLearningCycles", cfg.NumCycles, ...
        "LearnRate", cfg.LearnRate, "Learners", t);
end
