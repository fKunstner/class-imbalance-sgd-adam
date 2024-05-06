import warnings

from optexp.experiments.experiment import Experiment

_displayname = {
    "va_Accuracy": "Val. Accuracy",
    "val_Accuracy": "Val. Accuracy",
    "tr_Accuracy": "Train Accuracy",
    "va_CrossEntropyLoss": "Val. Loss",
    "val_CrossEntropyLoss": "Val. Loss",
    "tr_CrossEntropyLoss": "Train Loss",
    "va_ClassificationSquaredLoss": "Val. MSE",
    "val_ClassificationSquaredLoss": "Val. MSE",
    "tr_ClassificationSquaredLoss": "Train MSE",
    "tr_MSELoss": "Train MSE",
    "va_MSELoss": "Val. MSE",
    "val_MSELoss": "Val. MSE",
    "LogReg_BalancedXImbalancedY": "Balanced X, Imbalanced Y",
    "LogReg_BalancedXImbalancedY_PerClass": "Balanced X, Imbalanced Y",
    "LogReg_BalancedXImbalancedY_smaller_longer": "Small Balanced X, Imbalanced Y",
    "LogReg_Synthetic": "Synthetic Multiclass",
    "LogReg_Synthetic_Per_Class": "Synthetic Multiclass",
    "Dummy_LinReg": "Synthetic Regression",
    "Dummy_LinReg_Per_Class": "Synthetic Regression",
    "SimpleCNN_ImbalancedMNIST_FB_Base_PerClass": "MNIST Imbalanced",
    "SimpleCNN_MNIST_FB_Base": "CNN on MNIST",
    "SimpleCNN_ImbalancedMNIST_FB_Base": "CNN on Imbalanced MNIST",
    "TEnc_standard_training_PTB": "2-layer Transformer on PTB",
    "TransformerEncoder_TinyPTB_FB_Width_EmbDim_1200": "One layer",
    "TransformerEncoder_TinyPTB_FB_No_MLP_Norm_Drop_EmbDim_1200": "No Bells/Whistles",
    "TransformerEncoder_TinyPTB_FB_Base_TTF": "Last Layer only",
    "MLP_Frozen_split_by_class": "Layer Layer only",
    "MLP_Frozen_100_FB_Start_Config_0.08": "Layer Layer (Tiny)",
    "MLP_Frozen_100_FB_Start_Config_0.08_split_by_class": "Layer Layer (Tiny)",
    "TEnc_standard_training_PTB_per_class": "2-Layer transformer on PTB",
    "SimpleCNN_MNISTBarcodedOnly_FB_normalized_long": "BarcodedMNIST",
}


def displayname(x):
    if x not in _displayname:
        print(
            f"--------------------------- "
            f"Warning: Unknown displayname for {x} in displayname()"
        )
        return x
    return _displayname[x]


def metrics_to_plot_and_main_metric_for_standard_plots(problem):
    metric_names = []
    for criterion in problem.get_criterions():
        if "PerClass" not in criterion.__class__.__name__:
            metric_names.append(f"tr_{criterion.__class__.__name__}")
            metric_names.append(f"va_{criterion.__class__.__name__}")
            metric_names.append(f"val_{criterion.__class__.__name__}")
    main_metric = [
        _ for _ in metric_names if "loss" in _.lower() and "tr" in _.lower()
    ][0]
    return main_metric, metric_names


def get_ylims_for_problem_linear(exp: Experiment, key: str):
    if "Accuracy" in key:
        return [0, 1]

    if "PTB" in exp.problem.dataset.name:
        if "CrossEntropy" in key:
            return [0, 11]

    if "Dummy_LinReg" in exp.group:
        if "SquaredLoss" in key:
            return [0, 7]

    if "MNIST" in exp.group:
        if "CrossEntropy" in key:
            return [0, 11]

    if "LogReg_Synthetic" in exp.group:
        if "CrossEntropy" in key:
            return [0, 11]

    if "MLP_Frozen" in exp.group:
        if "CrossEntropy" in key:
            return [0, 11]

    if "BalancedXImbalancedY" in exp.group:
        if "CrossEntropy" in key:
            return [0, 11]

    warnings.warn(f"Unknown ylim combo for {key}, {exp}")
    return None


def get_ylims_for_problem_log(exp: Experiment, key: str):
    warnings.warn(f"Unknown ylim combo for {key}, {exp}")

    if "GPT2Small_Wikitext103_Per_Class_Stats_15k_Steps" in exp.group:
        if "Accuracy" in key:
            return [10**-6, 1]

    return None


def should_plot_logy(metric):
    if "Accuracy" in metric:
        return False
    return True
