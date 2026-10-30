%% Compare more Regression Learner app models on the same features and split
% Run after step 2. Uses the year 1-5 / year 6 split from preparedData.mat and
% adds Fine Tree, Medium Tree and Neural Network to step 2's results table,
% labelling every model with its Regression Learner app name.

clear; clc; close all
rng(0)
load("preparedData.mat")
load("bestModelConfig.mat", "results")

Xtr  = trainSet(:, predictorNames);  ytr  = trainSet.site_eui;
Xval = valSet(:, predictorNames);    yval = valSet.site_eui;
rmse = @(y, yhat) sqrt(mean((y - yhat).^2));
addResult = @(results, name, yhat) [results; ...
    {name, rmse(yval, yhat), rmse(yval(valTestLike), yhat(valTestLike))}];

%% Trees: missing values get the out-of-range sentinel, as in step 2
XtrT = fillSentinel(Xtr);  XvalT = fillSentinel(Xval);
results = addResult(results, "Fine Tree", predict(fitrtree(XtrT, ytr, "MinLeafSize", 4), XvalT));
results = addResult(results, "Medium Tree", predict(fitrtree(XtrT, ytr, "MinLeafSize", 12), XvalT));

%% Neural network: standardised inputs, so fill gaps with the training median
% (a -999 sentinel would dominate the scaling) and keep the missing flags as inputs.
XtrN = Xtr;  XvalN = Xval;
isNum = varfun(@isnumeric, Xtr, "OutputFormat", "uniform");
for v = predictorNames(isNum)
    med = median(Xtr.(v), "omitnan");
    XtrN.(v)  = fillmissing(Xtr.(v),  "constant", med);
    XvalN.(v) = fillmissing(Xval.(v), "constant", med);
end
XtrN.lag_missing = double(trainSet.lag_count == 0);
XvalN.lag_missing = double(valSet.lag_count == 0);
nnMdl = fitrnet(XtrN, ytr, "LayerSizes", 10, "Standardize", true);
results = addResult(results, "Neural Network (one layer, 10 units)", predict(nnMdl, XvalN));

%% Relabel step 2's models with their app names
appNames = dictionary( ...
    ["Mean of training data", "Linear regression", "Random forest", "Boosted trees", ...
     "Boosted trees, log target", "Boosted trees, tuned", "Average of forest and boosting"], ...
    ["Mean of training data", "Linear", "Bagged Trees (Random Forest)", "Boosted Trees", ...
     "Boosted Trees, log target", "Boosted Trees, tuned", ...
     "Average of Bagged Trees and Boosted Trees (two app models)"]);
known = isKey(appNames, results.Model);
results.Model(known) = appNames(results.Model(known));
results = sortrows(results, "RMSE_TestLike");
disp(results)
