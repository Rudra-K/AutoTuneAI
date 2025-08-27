def detect_framework(model):
    """
    Detects the model's framework using introspection.
    """
    module_path = model.__class__.__module__

    if module_path.startswith('sklearn.'):
        return "sklearn"
    elif module_path.startswith('torch.'):
        return "pytorch"
    elif module_path.startswith('tensorflow.') or module_path.startswith('keras.'):
        return "tensorflow"
    elif module_path.startswith('xgboost.'):
        return "xgboost"
    elif module_path.startswith('lightgbm.'):
        return "lightgbm"
    elif module_path.startswith('catboost.'):
        return "catboost"
    
    return "unknown"

def detect_model_type(model):
    """
    Detects the general model type, returning a key that matches
    the search_space_config.py file.
    """
    name = model.__class__.__name__.lower()

    # Deep Learning Models:
    if any(x in name for x in ["cnn", "resnet", "conv"]):
        return "cnn"
    if any(x in name for x in ["transformer", "bert", "gpt"]):
        return "transformer"
    if any(x in name for x in ["rnn", "lstm", "gru"]):
        return "rnn"
    
    # Tree-Based Models:
    if "randomforest" in name:
        return "random_forest"
    if "xgb" in name:
        return "xgboost"
    if "lgbm" in name:
        return "lightgbm"
    if "catboost" in name:
        return "catboost"
        
    # Other ML Models:
    if "logisticregression" in name:
        return "logistic_regression"
    if "svm" in name or "svc" in name:
        return "svm"

    return "unknown"

def detect_task_type(model, framework="unknown"):
    """
    Detects the model's task type .
    """
    if framework in ["sklearn", "xgboost", "lightgbm", "catboost"]:
        if hasattr(model, '_estimator_type'):
            if model._estimator_type == "classifier":
                return "classification"
            if model._estimator_type == "regressor":
                return "regression"

    name = model.__class__.__name__.lower()
    if any(x in name for x in ["resnet", "cnn", "vgg"]):
        return "cv"
    if any(x in name for x in ["bert", "gpt", "transformer"]):
        return "nlp"
    if any(x in name for x in ["rnn", "gru", "lstm"]):
        return "sequence"
    if "classifier" in name or "svc" in name:
        return "classification"
    if "regressor" in name:
        return "regression"
        
    return "unknown"