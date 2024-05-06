def pprint_dict(adict):
    """Pretty printing for logging dictionaries.

    Ignores keys containing lists, prints floating points in scientific
    notation, shows accuracy in percentage.
    """

    def fmt_entry(k, v):
        if isinstance(v, float):
            if "Accuracy" in k or "acc" in k:
                return f"{k}={100*v:.1f}"
            else:
                return f"{k}={v:.2e}"
        else:
            return f"{k}={v}"

    def filter_metric(k):
        x = "PerSequenceLength" not in k
        y = "norm" not in k
        return x and y

    return (
        "{"
        + ", ".join(
            fmt_entry(k, v)
            for k, v in sorted(adict.items())
            if not hasattr(v, "__len__") and filter_metric(k)
        )
        + "}"
    )
