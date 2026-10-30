%% Step 1: Load, clean and prepare the data
% Run this from the repo root; it reads data/train.csv and data/test.csv.
% It saves preparedData.mat, which step 2 (modelling) loads.

clear; clc; close all
rng(0)  % reproducible results

%% Load the data
train = readtable(fullfile("data", "train.csv"), "TextType", "string");
test  = readtable(fullfile("data", "test.csv"), "TextType", "string");
fprintf("Train: %d rows, Test: %d rows\n", height(train), height(test))

%% Explore: what is missing, and how does test differ from train?
commonVars = intersect(string(train.Properties.VariableNames), ...
    string(test.Properties.VariableNames), "stable");
pctMissingTrain = 100 * mean(ismissing(train(:, commonVars)));
pctMissingTest  = 100 * mean(ismissing(test(:, commonVars)));
hasGaps = pctMissingTrain > 0 | pctMissingTest > 0;
missingSummary = table(commonVars(hasGaps)', pctMissingTrain(hasGaps)', pctMissingTest(hasGaps)', ...
    'VariableNames', ["Variable", "PctMissingTrain", "PctMissingTest"]);
disp(missingSummary)

% Key findings that drive the choices below:
%  * Test is a single future year (Year_Factor = 7); train covers years 1-6.
%  * State_6 is ~67% of train but does not appear in test at all.
%  * The wind and fog columns are 54-60% missing in train and 88-94% in test.
%  * energy_star_rating is 35% missing in train; year_built has some 0 placeholders.
figure
tiledlayout(1, 2)
nexttile
histogram(train.site_eui, 100)
xlabel("Site EUI"); ylabel("Count"); title("Target distribution (right-skewed)")
nexttile
states = categorical([train.State_Factor; test.State_Factor]);
src = categorical([repmat("Train", height(train), 1); repmat("Test", height(test), 1)]);
counts = crosstab(states, src);
bar(categorical(categories(states)), counts ./ sum(counts), "grouped")
legend(categories(src)); ylabel("Share of rows"); title("State mix: train vs test")

%% Clean and engineer features (same steps for train and test)
train = cleanFeatures(train);
test  = cleanFeatures(test);

% Encode text columns as categoricals with the same categories in both sets
catVars = ["State_Factor", "building_class", "facility_type"];
for v = catVars
    allCats = unique([train.(v); test.(v)]);
    train.(v) = categorical(train.(v), allCats);
    test.(v)  = categorical(test.(v),  allCats);
end

%% Building history (lag) features
% Rows have no building id, but the same building reappears across years with
% identical static features. Its Site EUI in earlier years is the strongest
% predictor we have (the core of the WiDS 2022 winning approach).
[train, test] = addHistoryFeatures(train, test);
fprintf("Rows with building history: %.1f%% of train, %.1f%% of test\n", ...
    100 * mean(train.lag_count > 0), 100 * mean(test.lag_count > 0))

%% Choose predictors
% id identifies rows only, and Year_Factor is used for splitting: the test year (7)
% is outside the training range, so the model should not lean on it.
predictorNames = setdiff(string(train.Properties.VariableNames), ...
    ["id", "site_eui", "Year_Factor"], "stable");
fprintf("Using %d predictors\n", numel(predictorNames))

%% Validation split that mimics the test set
% The test set is a future year, so a random split would be too optimistic.
% Train on years 1-5 and validate on year 6. We also report a "test-like"
% validation score that leaves out State_6, since test has no State_6 buildings.
isVal = train.Year_Factor == 6;
trainSet = train(~isVal, :);
valSet   = train(isVal, :);
valTestLike = valSet.State_Factor ~= "State_6";
fprintf("Train split: %d rows, validation: %d rows (%d test-like)\n", ...
    height(trainSet), height(valSet), nnz(valTestLike))

save("preparedData.mat", "train", "test", "trainSet", "valSet", "valTestLike", "predictorNames")
disp("Saved preparedData.mat")

%% Local functions
function T = cleanFeatures(T)
    % Drop the wind and fog columns: mostly missing, and almost entirely missing in test
    T = removevars(T, intersect(string(T.Properties.VariableNames), ...
        ["direction_max_wind_speed", "direction_peak_wind_speed", "max_wind_speed", "days_with_fog"]));

    % year_built of 0 is a placeholder, not a real year
    T.year_built(T.year_built < 1600) = NaN;

    % Missing Energy Star rating is informative in itself (unrated buildings
    % differ from rated ones), so flag it. Tree models handle the NaN directly.
    T.energy_star_missing = double(isnan(T.energy_star_rating));

    % Floor area spans several orders of magnitude
    T.log_floor_area = log(T.floor_area);

    % Seasonal temperature summaries (smoother than 36 separate monthly columns)
    T.winter_avg_temp = mean([T.december_avg_temp, T.january_avg_temp, T.february_avg_temp], 2);
    T.summer_avg_temp = mean([T.june_avg_temp, T.july_avg_temp, T.august_avg_temp], 2);
    T.annual_temp_range = T.summer_avg_temp - T.winter_avg_temp;
end

function [train, test] = addHistoryFeatures(train, test)
    % Identify each building by its static features, then, for every row, summarise
    % that building's Site EUI in strictly earlier years (so nothing leaks from the
    % row's own year or the future, for train, validation and test alike).
    keyVars = ["State_Factor", "building_class", "facility_type", "floor_area", "year_built"];
    both = [train(:, [keyVars, "Year_Factor"]); test(:, [keyVars, "Year_Factor"])];
    yearBuilt = fillmissing(both.year_built, "constant", -1);  % keep unknown years as one key value
    bld = findgroups(both.State_Factor, both.building_class, both.facility_type, ...
        both.floor_area, yearBuilt);
    yr = both.Year_Factor;
    eui = [train.site_eui; nan(height(test), 1)];

    % One Site EUI per building and year (a few keys cover several buildings)
    known = ~isnan(eui);
    byYear = groupsummary(table(bld(known), yr(known), eui(known), ...
        'VariableNames', ["bld", "yr", "eui"]), ["bld", "yr"], "mean", "eui");
    byYear = sortrows(byYear, ["bld", "yr"]);

    n = height(both);
    lagLast = nan(n, 1);  lagMean = nan(n, 1);  lagCount = zeros(n, 1);  lagGap = nan(n, 1);
    for y = unique(yr)'
        past = byYear(byYear.yr < y, :);
        if isempty(past), continue; end
        [pastBld, lastIdx] = unique(past.bld, "last");
        stats = groupsummary(past, "bld", "mean", "mean_eui");
        [isKnown, loc] = ismember(bld, pastBld);
        rows = yr == y & isKnown;
        lagLast(rows)  = past.mean_eui(lastIdx(loc(rows)));
        lagGap(rows)   = y - past.yr(lastIdx(loc(rows)));
        lagMean(rows)  = stats.mean_mean_eui(loc(rows));
        lagCount(rows) = stats.GroupCount(loc(rows));
    end

    isTrain = (1:n)' <= height(train);
    train.lag_last = lagLast(isTrain);   test.lag_last = lagLast(~isTrain);
    train.lag_mean = lagMean(isTrain);   test.lag_mean = lagMean(~isTrain);
    train.lag_count = lagCount(isTrain); test.lag_count = lagCount(~isTrain);
    train.lag_gap = lagGap(isTrain);     test.lag_gap = lagGap(~isTrain);
end
