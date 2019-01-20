import numpy as np
import pandas

def normalize(data, type_ = None, q_lower = None, q_upper = None, ref = None, A = None, b = None):
    """

    Normalization function

    Can be of type:
    - min-max: requires no arguments
    - quantile: requires a lower(q_lower) and upper(q_upper) quantile
    - ref: requires a reference value(ref)

    If no type is given, it returns a list normalized to the previous value

    """

    normed_df = pandas.DataFrame()

    if type_ == "min-max":
        x_min = data.values.min()
        x_max = data.values.max()
        normed_df = (data - x_min) / (x_max - x_min)

    elif type_ == "quantile":
        x_lower = np.percentile(data, q_lower * 100)
        x_upper = np.percentile(data, q_upper * 100)
        normed_df = (data - x_lower) / (x_upper- x_lower)

    elif type_ == "ref":
        normed_df = (data - ref) / ref

    elif type_ == "Ab":
        normed_df = A * data + b


    return normed_df