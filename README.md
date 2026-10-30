
 # Hackathon2026 MATLAB Hackathon 2026 by CCEB ANATA, held on 30/10/2026
  
 Team members: 
 
 1. Duong Minh Triet 
 2. Nguyen Dang Quang 
 3. Huynh Xuan Lam 
 4. Vuu Tan Loc
  
 ## Predicting building energy use (WiDS Datathon 2022 task)
  
 The task is to predict **Site EUI** (Site Energy Usage Intensity: the energy a building uses per square foot per year) for 9,705 buildings in a future year. The inputs are building characteristics and the local weather. Predictions are scored by RMSE (root mean squared error): roughly the typical size of a prediction error, in Site EUI units, with large misses weighted heavily.
  
 **Result:** a validation RMSE of **25.99** on test-like data, down from 64.88 for predicting the average and 46.84 for our best model without building history. Every model used is available in MATLAB's Regression Learner app.
  
 ## Data
  
 | File | Rows | Contents | |---|---|---| | `data/train.csv` | 75,757 | One building in one year, years 1-6, with the target `site_eui` | | `data/test.csv` | 9,705 | Year 7 only, no target | | `data/sample_solution.csv` | 9,705 | Submission format: `id,site_eui` |
  
 Each row has the building's state, class, facility type, floor area, year built, Energy Star rating, and elevation. It also has that year's weather: monthly min/avg/max temperatures, degree days, precipitation, snow, days above/below temperature thresholds, wind and fog.
  
 ## How to run
  
 Requirements: MATLAB R2025b (or a recent release) with the Statistics and Machine Learning Toolbox.
  
 From the repo root:
  
 ```matlab runAll ```
  
 Or from a terminal: `matlab -batch "runAll"`. A full run takes about 15-25 minutes and writes `submission.csv`. Intermediate `.mat` files and logs are git-ignored.
  
 ## Repository layout
  
 | File | Purpose | |---|---| | `runAll.m` | Runs steps 1-3 in order | | `step1_prepare_data.m` | Loads, explores and cleans the data, engineers features, adds building history, makes the validation split. Saves `preparedData.mat` | | `step2_train_models.m` | Trains and compares models on years 1-5 vs year 6, tunes boosting, picks the final approach. Saves `bestModelConfig.mat` | | `step3_make_submission.m` | Retrains the chosen approach on all six years, sanity-checks and writes `submission.csv` | | `trainForest.m` | Bagged Trees with predictor sampling (a random forest) | | `trainBoost.m`, `predictBoost.m` | LSBoost Boosted Trees, optionally on log(Site EUI) | | `fillSentinel.m` | Replaces missing numbers with -999 for tree models | | `compare_app_models.m` | Optional, run after step 2: adds Fine Tree, Medium Tree and Neural Network to the comparison | | `cleaned_data.m`, `test.m` | Earlier team exploration scripts | | `submission.csv` | Final predictions |
  
 ## Approach and thinking flow
  
 ### 1. Explore what makes the test set different
  
 | Finding | Decision | |---|---| | Test is a single future year (7); train is years 1-6 | `Year_Factor` is not a predictor; validate on the latest year | | State_6 is 67% of train and absent from test | Choose models on a "test-like" score that excludes State_6 | | Wind and fog columns are 88-94% missing in test | Drop those four columns | | Energy Star rating is 35% missing in train | Add an `energy_star_missing` flag | | `year_built` has 0 placeholders | Treat values below 1600 as missing | | Site EUI is long-tailed (mean 82.6, values up to ~1,000) | Try a log target as well |
  
 ### 2. Validate like the test set
  
 Train on years 1-5 and validate on year 6. Year 6 is scored twice: on all states, and "test-like" (9,432 rows without State_6). A random 80/20 split would let the model see the year it is scored on and give over-optimistic results. For the submission, the chosen model is retrained on all six years.
  
 ### 3. Engineer features
  
 - `log_floor_area`, because floor area spans several orders of magnitude. - `winter_avg_temp`, `summer_avg_temp` and `annual_temp_range`, which summarise the 36 monthly temperatures. - `energy_star_missing`, because unrated buildings behave differently.
  
 ### 4. Add building history: the biggest improvement
  
 The data has no building id, but a building keeps the same state, class, facility type, floor area and year built every year. Using that combination as a key, each row gets the building's Site EUI from **strictly earlier years**, so no information leaks from the year being predicted:
  
 - `lag_last`: Site EUI in the most recent earlier year - `lag_mean`: average Site EUI over all earlier years - `lag_count`: how many earlier years were observed - `lag_gap`: years since the last observation
  
 93.4% of test rows have history in the training data. This is the same core idea the first-place WiDS 2022 team built on (lag features). The core of `addHistoryFeatures` in `step1_prepare_data.m`:
  
 ```matlab keyVars = ["State_Factor", "building_class", "facility_type", "floor_area", "year_built"]; both = [train(:, [keyVars, "Year_Factor"]); test(:, [keyVars, "Year_Factor"])]; yearBuilt = fillmissing(both.year_built, "constant", -1);  % keep unknown years as one key value bld = findgroups(both.State_Factor, both.building_class, both.facility_type, ...     both.floor_area, yearBuilt); ... for y = unique(yr)'     past = byYear(byYear.yr < y, :);          % only years before this row's year     [pastBld, lastIdx] = unique(past.bld, "last");     stats = groupsummary(past, "bld", "mean", "mean_eui");     [isKnown, loc] = ismember(bld, pastBld);     rows = yr == y & isKnown;     lagLast(rows)  = past.mean_eui(lastIdx(loc(rows)));     lagGap(rows)   = y - past.yr(lastIdx(loc(rows)));     lagMean(rows)  = stats.mean_mean_eui(loc(rows));     lagCount(rows) = stats.GroupCount(loc(rows)); end ```
  
 The final model uses 66 predictors: 57 original columns, 5 engineered features and 4 history features.
  
 ### 5. Handle missing values honestly
  
 Without surrogate splits, MATLAB trees stop an observation with a missing value at the split node. Tree models therefore see missing numbers as -999, a value no real input takes, so "missing" can get its own branch. The linear model uses median imputation instead.
  
 ```matlab function X = fillSentinel(X)     isNum = varfun(@isnumeric, X, "OutputFormat", "uniform");     for v = string(X.Properties.VariableNames(isNum))         X.(v) = fillmissing(X.(v), "constant", -999);     end end ```
  
 ### 6. Compare models and pick from the evidence
  
 Step 2 trains baselines, Bagged Trees and Boosted Trees, and tunes boosting over a small grid. The grid uses a learning rate of 0.1, 127 or 255 splits, leaf size 10 or 30, and picks the number of trees from the validation curve. It then keeps whichever of Bagged Trees, tuned Boosted Trees or their average scores best on the test-like validation. Averaging two different tree ensembles cancels part of their errors.
  
 ## Results
  
 RMSE on year 6 (lower is better). Test-like = without State_6, which matches the test set.
  
 | Model (Regression Learner name) | All states | Test-like | |---|---|---| | **Average of Bagged Trees and Boosted Trees (final)** | 31.92 | **25.99** | | Boosted Trees, tuned (127 splits, leaf 10, 38 trees) | 33.60 | 26.35 | | Bagged Trees (Random Forest, 100 trees) | 31.42 | 26.35 | | Boosted Trees, default settings | 36.92 | 27.15 | | Boosted Trees, log target | 33.77 | 27.94 | | Medium Tree | 36.72 | 28.79 | | Fine Tree | 37.65 | 31.01 | | Linear | 36.10 | 35.83 | | Neural Network (one layer, 10 units) | 47.80 | 50.60 | | Mean of training data | 57.21 | 64.88 |
  
 Effect of building history, test-like RMSE:
  
 | Model | Without history | With history | |---|---|---| | Bagged Trees | 47.02 | 26.35 | | Boosted Trees, tuned | 48.51 | 26.35 | | Average of both | 46.84 | 25.99 | | Linear | 54.26 | 35.83 |
  
 Most important predictors in the final models (share of predictor importance):
  
 | Rank | Boosted Trees | Bagged Trees | |---|---|---| | 1 | `lag_last` 42% | `lag_last` 27% | | 2 | `facility_type` 19% | `lag_mean` 21% | | 3 | `lag_mean` 16% | `facility_type` 18% | | 4 | `energy_star_rating` 10% | `energy_star_rating` 10% | | 5 | `lag_gap` 2% | `log_floor_area` 4% |
  
 A building's own past energy use and its facility type explain most of the prediction. Weather adds comparatively little once history is known.
  
 ## Submission
  
 `submission.csv` has 9,705 rows in the format `id,site_eui`, in the same order as `data/sample_solution.csv`. All predictions are finite and at least 1 (mean 72.4, median 61.2).
  
 ## Possible next steps
  
 - Group averages, such as mean Site EUI per facility type or state, computed out of year. - Separate models for buildings with and without history. - More tuning of the tree count and leaf size, and more models in the average.
