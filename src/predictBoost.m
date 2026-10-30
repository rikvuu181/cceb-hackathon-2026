function yhat = predictBoost(mdl, X, cfg)
% Predict Site EUI, undoing the log transform if the model was trained on log(y).
    yhat = predict(mdl, X);
    if cfg.LogTarget
        yhat = exp(yhat);
    end
end
