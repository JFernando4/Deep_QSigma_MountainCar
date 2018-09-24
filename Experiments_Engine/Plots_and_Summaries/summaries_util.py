from scipy.stats import t, chi2   # Faster than importing scipy.stats
import numpy as np
import os

""" Summary Functions """


# Confidence Interval for averages
def compute_tdist_confidence_interval(sample_mean, sample_std, proportion, sample_size):
    if sample_size <= 1:
        return None, None, None

    dof = sample_size - 1
    t_dist = t(df=dof)  # from scipy.stats
    tdist_factor = t_dist.ppf(1 - proportion/2)

    sqrt_inverse_sample_size = np.sqrt(1 / sample_size)
    me = sample_std * tdist_factor * sqrt_inverse_sample_size
    upper_bound = sample_mean + me
    lower_bound = sample_mean - me

    return upper_bound, lower_bound, me


# Confidence Interval for variance
def compute_chi2dist_confidence_interval(sample_variance, proportion, sample_size):
    dof = sample_size - 1
    chi2_dist = chi2(dof)

    upper_proportion = 1 - (proportion / 2)
    lower_proportion = proportion / 2

    upper_chi2_value = chi2_dist.ppf(upper_proportion)
    lower_chi2_value = chi2_dist.ppf(lower_proportion)

    # the lower and upper chi2_values get inverted because we're bounding sigma^2 in the equation:
    # chi2.ppf(proportion/2) < (n-1) s^2 / sigma^2 < chi2.ppf(1 - proportion/2)
    upper_bound_ci = (dof * sample_variance) / lower_chi2_value
    lower_bound_ci = (dof * sample_variance) / upper_chi2_value

    return upper_bound_ci, lower_bound_ci


# Create Results File
def create_results_file(pathname, columns, headers, title="", addtofile=False, results_name="results"):
    numrows = len(columns[0])
    numcolumns = len(columns)
    if addtofile: assert numcolumns == len(headers)
    assert all(len(acolumn) == numrows for acolumn in columns)

    results_path = os.path.join(pathname, results_name+".txt")

    mode = 'a'
    if addtofile:
        mode = 'a'
    with open(results_path, mode=mode) as results_file:
        # if not addtofile:
        results_file.write("\n##########  " + title + "  ##########\n\n")
        for header in headers:
            results_file.write(header + "\t")
        results_file.write('\n')
        for j in range(numrows):
            for i in range(numcolumns):
                results_file.write(str(columns[i][j]) + "\t")
            results_file.write("\n")
    return
