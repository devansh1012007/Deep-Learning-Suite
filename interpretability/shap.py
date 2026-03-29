import shap

def explain_model(model, data):
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)
    return shap_values