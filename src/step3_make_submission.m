%% Step 3: Retrain the chosen approach on all data and write the submission file
% Needs preparedData.mat (step 1) and bestModelConfig.mat (step 2).
% Writes submission.csv in the same id,site_eui format as sample_solution.csv.

clear; clc; close all
rng(0)
load("preparedData.mat")
load("bestModelConfig.mat")

%% Retrain the chosen approach on all six training years and predict the test set
% Year 6 was held out for validation; now it is the most recent and most
% useful data we have, so the final model learns from it too.
Xall = train(:, predictorNames);  yall = train.site_eui;
Xall = fillSentinel(Xall);
Xtest = fillSentinel(test(:, predictorNames));
fprintf("Retraining: %s\n", finalChoice)
switch finalChoice
    case "Random forest"
        yhatTest = predict(trainForest(Xall, yall, rfCfg), Xtest);
    case "Boosted trees, tuned"
        yhatTest = predictBoost(trainBoost(Xall, yall, bestCfg), Xtest, bestCfg);
    otherwise  % average of both
        yhatTest = (predict(trainForest(Xall, yall, rfCfg), Xtest) + ...
            predictBoost(trainBoost(Xall, yall, bestCfg), Xtest, bestCfg)) / 2;
end
yhatTest = max(yhatTest, 1);  % Site EUI is always positive (training minimum is ~1)

submission = table(test.id, yhatTest, 'VariableNames', ["id", "site_eui"]);

%% Sanity checks before saving
sample = readtable(fullfile("data", "sample_solution.csv"));
assert(isequal(sample.id, submission.id), "Row order does not match sample_solution.csv")
assert(all(isfinite(submission.site_eui)), "Some predictions are NaN or Inf")

writetable(submission, "submission.csv")
fprintf("Wrote submission.csv with %d predictions (mean %.1f, median %.1f)\n", ...
    height(submission), mean(yhatTest), median(yhatTest))

figure
histogram(train.site_eui(train.State_Factor ~= "State_6"), 0:5:400, "Normalization", "pdf")
hold on
histogram(yhatTest, 0:5:400, "Normalization", "pdf")
hold off
legend("Training actuals (excluding State 6)", "Test predictions")
xlabel("Site EUI"); ylabel("Density"); title("Do the test predictions look plausible?")
