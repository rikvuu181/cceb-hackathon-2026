%% Step 2: Train and compare models
% Needs preparedData.mat from step1_prepare_data.m, plus trainForest.m, trainBoost.m, predictBoost.m and fillSentinel.m.
% Trains on years 1-5, scores on year 6, and saves the winning settings to
% bestModelConfig.mat for step 3.

clear; clc; close all
rng(0)
load("preparedData.mat")

Xtr  = trainSet(:, predictorNames);  ytr  = trainSet.site_eui;
Xval = valSet(:, predictorNames);    yval = valSet.site_eui;

rmse = @(y, yhat) sqrt(mean((y - yhat).^2));
results = table('Size', [0 3], 'VariableTypes', ["string", "double", "double"], ...
    'VariableNames', ["Model", "RMSE_Val", "RMSE_TestLike"]);
addResult = @(results, name, yhat) [results; ...
    {name, rmse(yval, yhat), rmse(yval(valTestLike), yhat(valTestLike))}];

%% Baseline 1: predict the training mean for every building
results = addResult(results, "Mean of training data", repmat(mean(ytr), size(yval)));

%% Baseline 2: linear regression
% Linear models cannot use NaN, so fill numeric gaps with the training median.
isNum = varfun(@isnumeric, Xtr, "OutputFormat", "uniform");
XtrLin = Xtr;  XvalLin = Xval;
for v = predictorNames(isNum)
    med = median(Xtr.(v), "omitnan");
    XtrLin.(v)  = fillmissing(Xtr.(v),  "constant", med);
    XvalLin.(v) = fillmissing(Xval.(v), "constant", med);
end
XtrLin.site_eui = ytr;
linMdl = fitlm(XtrLin, "ResponseVar", "site_eui");
yhatLin = predict(linMdl, XvalLin);
yhatLin(isnan(yhatLin)) = mean(ytr);  % facility types never seen in years 1-5
results = addResult(results, "Linear regression", yhatLin);

% Tree models get an out-of-range value for missing numbers instead (see fillSentinel.m)
Xtr = fillSentinel(Xtr);  Xval = fillSentinel(Xval);

%% Random forest (bagged trees, each split sees a third of the predictors)
rfCfg = struct("NumTrees", 100, "MinLeafSize", 5, ...
    "NumVariablesToSample", round(numel(predictorNames) / 3));
rfMdl = trainForest(Xtr, ytr, rfCfg);
yhatRF = predict(rfMdl, Xval);
results = addResult(results, "Random forest", yhatRF);

%% Gradient-boosted trees
cfg = struct("LearnRate", 0.1, "MaxNumSplits", 63, "MinLeafSize", 20, ...
    "NumCycles", 300, "LogTarget", false);
boostMdl = trainBoost(Xtr, ytr, cfg);
results = addResult(results, "Boosted trees", predictBoost(boostMdl, Xval, cfg));

% Same model on log(site_eui): the target is right-skewed with values up to ~1000
cfgLog = cfg;  cfgLog.LogTarget = true;
boostLogMdl = trainBoost(Xtr, ytr, cfgLog);
results = addResult(results, "Boosted trees, log target", predictBoost(boostLogMdl, Xval, cfgLog));

disp(results)

%% Tune the boosted model with a small grid
% A grid keeps run time predictable; each fit takes about a minute.
% Earlier runs favoured deep trees and a learning rate of 0.1, so the grid stays there.
[lr, splits, leaf] = ndgrid(0.1, [127 255], [10 30]);
grid = table(lr(:), splits(:), leaf(:), ...
    'VariableNames', ["LearnRate", "MaxNumSplits", "MinLeafSize"]);
grid.RMSE_TestLike = nan(height(grid), 1);
grid.BestCycles = nan(height(grid), 1);
for i = 1:height(grid)
    c = cfg;
    c.LearnRate = grid.LearnRate(i);
    c.MaxNumSplits = grid.MaxNumSplits(i);
    c.MinLeafSize = grid.MinLeafSize(i);
    c.NumCycles = 500;
    mdl = trainBoost(Xtr, ytr, c);
    % Validation error after each added tree, so we also pick how many trees to keep
    curve = sqrt(loss(mdl, Xval(valTestLike, :), yval(valTestLike), "Mode", "cumulative"));
    [grid.RMSE_TestLike(i), grid.BestCycles(i)] = min(curve);
    fprintf("Grid %d/%d: RMSE %.2f at %d trees\n", i, height(grid), ...
        grid.RMSE_TestLike(i), grid.BestCycles(i))
end
grid = sortrows(grid, "RMSE_TestLike");
disp(grid)

bestCfg = cfg;
bestCfg.LearnRate = grid.LearnRate(1);
bestCfg.MaxNumSplits = grid.MaxNumSplits(1);
bestCfg.MinLeafSize = grid.MinLeafSize(1);
bestCfg.NumCycles = grid.BestCycles(1);
bestMdl = trainBoost(Xtr, ytr, bestCfg);
yhatBest = predictBoost(bestMdl, Xval, bestCfg);
results = addResult(results, "Boosted trees, tuned", yhatBest);

%% Pick the final approach: forest, tuned boosting, or the average of both
% Averaging two different tree ensembles often cancels part of their errors.
results = addResult(results, "Average of forest and boosting", (yhatRF + yhatBest) / 2);
disp(results)
candidates = ["Random forest", "Boosted trees, tuned", "Average of forest and boosting"];
candRows = find(ismember(results.Model, candidates));
[~, k] = min(results.RMSE_TestLike(candRows));
finalChoice = results.Model(candRows(k));
fprintf("Final choice for the submission: %s\n", finalChoice)

%% Visualise the results
figure
tiledlayout(1, 3)

nexttile
bar(categorical(results.Model, results.Model), [results.RMSE_Val, results.RMSE_TestLike])
ylabel("RMSE (Site EUI)"); legend("Year 6, all states", "Year 6, test-like")
title("Model comparison")

nexttile
scatter(yval(valTestLike), yhatBest(valTestLike), 4, "filled", "MarkerFaceAlpha", 0.3)
hold on; plot([0 600], [0 600], "k--"); hold off
xlim([0 600]); ylim([0 600])
xlabel("Actual Site EUI"); ylabel("Predicted"); title("Tuned model, test-like validation")

nexttile
imp = predictorImportance(bestMdl);
[imp, order] = sort(imp, "descend");
top = min(15, numel(imp));
barh(categorical(predictorNames(order(1:top)), flip(predictorNames(order(1:top)))), imp(1:top))
title("Top 15 predictors")

save("bestModelConfig.mat", "bestCfg", "rfCfg", "finalChoice", "results", "grid")
disp("Saved bestModelConfig.mat")
