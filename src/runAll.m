%% Run the full pipeline: data prep, model comparison, submission
% Run from the repo root; the CSVs are read from data/.

run("step1_prepare_data.m")
run("step2_train_models.m")
run("step3_make_submission.m")
