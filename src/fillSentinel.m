function X = fillSentinel(X)
% Replace NaN in numeric predictors with an out-of-range value for tree models.
% MATLAB trees without surrogate splits stop a NaN observation at the split node,
% so an explicit sentinel lets the tree send "missing" down its own branch.
    isNum = varfun(@isnumeric, X, "OutputFormat", "uniform");
    for v = string(X.Properties.VariableNames(isNum))
        X.(v) = fillmissing(X.(v), "constant", -999);
    end
end
