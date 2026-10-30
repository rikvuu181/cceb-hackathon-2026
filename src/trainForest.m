function mdl = trainForest(X, y, cfg)
% Fit a random forest: bagged regression trees where each split considers
% a random subset of predictors. cfg fields: NumTrees, MinLeafSize, NumVariablesToSample.
    t = templateTree("MinLeafSize", cfg.MinLeafSize, "NumVariablesToSample", cfg.NumVariablesToSample);
    mdl = fitrensemble(X, y, "Method", "Bag", "NumLearningCycles", cfg.NumTrees, "Learners", t);
end
