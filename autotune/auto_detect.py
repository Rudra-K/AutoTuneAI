import torch.nn as nn

def detect_model_type(model):
    name = model.__class__.__name__.lower()
    if any(x in name for x in ["cnn", "resnet", "conv"]):
        return "cnn"
    elif any(x in name for x in ["transformer", "bert", "gpt"]):
        return "transformer"
    elif any(x in name for x in ["rnn", "lstm", "gru"]):
        return "rnn"
    elif any(x in name for x in ["dqn", "ppo", "a2c"]):
        return "rl"
    elif "logisticregression" in name:
        return "linear"
    elif "randomforest" in name:
        return "tree"
    elif "svm" in name or "svc" in name:
        return "svm"
    elif "xgb" in name or "xgboost" in name:
        return "boost"
    return "unknown"

def detect_task_type(model):
    name = model.__class__.__name__.lower()
    if any(x in name for x in ["resnet", "cnn", "vgg"]):
        return "cv"
    elif any(x in name for x in ["bert", "gpt", "transformer"]):
        return "nlp"
    elif any(x in name for x in ["rnn", "gru", "lstm"]):
        return "sequence"
    elif any(x in name for x in ["dqn", "ppo", "a2c"]):
        return "rl"
    elif "logisticregression" in name or "svc" in name:
        return "classification"
    elif "linearregression" in name or "ridge" in name or "lasso" in name:
        return "regression"
    elif "randomforest" in name or "xgb" in name:
        return "classification"  # You can later refine using model._estimator_type if needed
    return "unknown"

def detect_framework(model):
    import importlib

    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return "pytorch"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return "tensorflow"
    except ImportError:
        pass

    try:
        import sklearn.base
        if isinstance(model, sklearn.base.BaseEstimator):
            return "sklearn"
    except ImportError:
        pass

    try:
        import xgboost as xgb
        if isinstance(model, xgb.XGBModel):
            return "xgboost"
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        if isinstance(model, lgb.LGBMModel):
            return "lightgbm"
    except ImportError:
        pass

    try:
        import catboost
        if isinstance(model, catboost.CatBoost):
            return "catboost"
    except ImportError:
        pass

    return "unknown"

