from optexp.experiments.big_experiments import dispatch_cmdline_to_all_groups

if __name__ == "__main__":
    ##
    # Ablation experiments reproducing a gap in performance between Adam and GD
    # on a subset of TinyPTB (PTB dataset, but training on the validation set)
    # with simplified Transformer models.
    #
    # The experiments are:
    # - basic_one_layer: One transformer module
    # - basic_one_layer_perclass: One transformer module, logging the per-class data
    # - no_mlp_layernorm_dropout: Removing the MLP, nonlinearity, LayerNorm, and dropout
    # - train_only_last_layer: Only training the last layer
    # - train_only_last_layer_perclass: Only training the last layer,
    #   logging the per-class data
    # - train_only_last_layer_perclass_tiny: Only training the last layer,
    #   logging the per-class data, with a small model (embedding dimension 100)
    #   and a small dataset (8% of the validation set)

    # Note: In the library, the "Frozen" dataset corresponds to the feature
    # transformation of a one-module transformer with embedding dimension 1200.
    # The "Frozen_{embedding_dimension}" datasets use other embedding dimensions.

    from optexp.experiments.simpler_transformers import (
        basic_one_layer,
        basic_one_layer_perclass,
        train_only_last_layer,
        train_only_last_layer_perclass,
    )

    groups = [
        basic_one_layer,
        basic_one_layer_perclass,
        train_only_last_layer,
        train_only_last_layer_perclass,
    ]

    ##
    # Experiments establishing the class imbalance issue when training
    # a 2-layer transformer with bells and whistles;
    # using LayerNorm, dropout, non-linearities in small batch training.
    #
    # The experiments are:
    # - PTB: Grid-search
    # - PTB_class_weighted: Class-weighted loss
    # - PTB_class_weighted_per_class: Logging the per-class data
    # - PTB_logit_adjusted: Adjusting the logits
    # - PTB_with_class_stats: Logging the per-class data
    # - PTB_with_class_stats_alt_opt: Additional optimizers

    from optexp.experiments.imbalance import (
        PTB,
        PTB_class_weighted,
        PTB_class_weighted_per_class,
        PTB_with_class_stats,
        PTB_with_class_stats_alt_opt,
    )

    groups += [
        PTB,
        PTB_class_weighted,
        PTB_class_weighted_per_class,
        PTB_with_class_stats,
        PTB_with_class_stats_alt_opt,
    ]

    ##
    # Experiments replicating the class imbalance with a CNN on vision data
    #
    # The experiments are:
    # - mnist_only: Standard MNIST
    # - mnist_barcoded_only: MNIST with barcodes
    # - mnist_barcoded_only_long: MNIST with barcodes
    # - mnist_and_barcoded: MNIST + MNIST with barcodes
    # - mnist_and_barcoded_perclass: Logging the per-class data
    # - mnist_and_barcoded_long: MNIST + MNIST with barcodes
    # - mnist_and_barcoded_perclass_long: Logging the per-class data
    # - mnist_and_barcoded_reweighted: Reweighting the loss
    from optexp.experiments.vision import mnist_only
    from optexp.experiments.vision.barcoded import (
        mnist_and_barcoded,
        mnist_and_barcoded_long,
        mnist_and_barcoded_long_perclass,
        mnist_and_barcoded_perclass,
        mnist_and_barcoded_reweighted,
        mnist_and_barcoded_reweighted_long,
        mnist_and_barcoded_reweighted_sqrt,
        mnist_barcoded_only,
        mnist_barcoded_only_long,
    )

    groups += [
        mnist_and_barcoded,
        mnist_and_barcoded_long,
        mnist_and_barcoded_long_perclass,
        mnist_and_barcoded_perclass,
        mnist_and_barcoded_reweighted,
        mnist_and_barcoded_reweighted_long,
        mnist_and_barcoded_reweighted_sqrt,
        mnist_barcoded_only,
        mnist_barcoded_only_long,
        mnist_only,
    ]

    ##
    # Toy models replicating the effect of class imbalance
    #
    # The experiments are:
    # - balanced_x: Logistic regression with balanced X data and imbalanced Y
    # - balanced_x_perclass: Logging the per-class data
    # - linreg: Linear regression learning an identity function with imbalanced data
    # - linreg_perclass: Logging the per-class data
    # - logreg: Softmax regression with data sampled from a cube
    # - logreg_perclass: Logging the per-class data

    from optexp.experiments.toy_models import (
        balanced_x,
        balanced_x_perclass,
        balanced_x_perclass_alt_opt,
    )

    groups += [
        balanced_x,
        balanced_x_perclass,
        balanced_x_perclass_alt_opt,
    ]

    # from optexp.experiments.toy_models.unused import (
    #     linreg,
    #     linreg_perclass,
    #     logreg,
    #     logreg_perclass,
    # )

    # groups += [
    #     linreg,
    #     linreg_perclass,
    #     logreg,
    #     logreg_perclass,
    # ]

    dispatch_cmdline_to_all_groups(groups)
