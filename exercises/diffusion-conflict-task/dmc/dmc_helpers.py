"""
Code written by Simon Schaefer.
Source: https://github.com/simschaefer/amortized-dmc
"""

import pandas as pd
import numpy as np
import time
import bayesflow as bf
from dmc import DMC
import copy
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Mapping, Sequence, Union, Dict, List, Any, Iterable, Hashable, Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy.typing as npt
from tqdm import tqdm


def hdi(
    samples: Sequence[float] | npt.NDArray[np.floating],
    hdi_prob: float = 0.95
) -> Tuple[float, float]:
    """
    Compute the Highest Density Interval (HDI) of a sample distribution.

    Parameters
    ----------
    samples : Sequence[float] or numpy.ndarray
        1D array-like object of posterior samples.
    hdi_prob : float
        The desired probability for the HDI (e.g., 0.95 for 95% HDI).

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds of the HDI.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("Only 1D arrays are supported.")
    
    sorted_samples = np.sort(samples)
    n_samples = len(sorted_samples)
    interval_idx_inc = int(np.floor(hdi_prob * n_samples))
    n_intervals = n_samples - interval_idx_inc

    if n_intervals <= 0:
        raise ValueError("Not enough samples for the desired HDI probability.")

    intervals = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(intervals)

    hdi_min = float(sorted_samples[min_idx])
    hdi_max = float(sorted_samples[min_idx + interval_idx_inc])

    return hdi_min, hdi_max


def check_vars(data: pd.DataFrame,
               rt: str = None,
               accuracy: str = None,
               id_name: str = None,
               congruency: str = None):

    var_names = ['Reaction Times (rt=...)', 'Accuracy (accuracy=...)', 'the Identifier (id_name=...)', 'Congruency (congruency=...)']

    for i, var in enumerate([rt, accuracy, id_name, congruency]):
        if var is not None:
            if var not in set(data.columns):
                raise ValueError(f"Variable '{var}' does not exist in data. Please specify a valid name for {var_names[i]}.")

def check_congruency(
    data: pd.DataFrame,
    rt: str = None,
    congruency: str = None,
    output_coding_con="congruent",
    output_coding_inc="incongruent",
):
    check_vars(data=data, rt=rt, congruency=congruency, accuracy=None, id_name=None)

    if congruency is None:
        return data

    congruency_labels = set(data[congruency].dropna().unique())

    allowed = [
        {"congruent", "incongruent"},
        {0, 1},
        {"con", "inc"},
        {output_coding_con, output_coding_inc},
    ]
    if not any(congruency_labels == s for s in allowed):
        raise ValueError(
            f"Congruency variable is coded as {congruency_labels}. Please recode "
            f"'{congruency}' to 'congruent' / 'incongruent' before submitting data to this function."
        )

    # recode using map (no FutureWarning)
    if congruency_labels == {"con", "inc"}:
        mapping = {"con": output_coding_con, "inc": output_coding_inc}
        data[congruency] = data[congruency].map(mapping).astype("object")

        mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
        mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
        diff = mean_inc - mean_con

        warnings.warn(
            f"'{congruency}' has been recoded to con -> {output_coding_con} / inc -> {output_coding_inc}. "
            f"RT Difference between incongruent - congruent conditions: {diff}."
        )

    elif congruency_labels == {0, 1} and congruency_labels != {output_coding_con, output_coding_inc}:
        mapping = {0: output_coding_con, 1: output_coding_inc}
        data[congruency] = data[congruency].map(mapping).astype("object")

        mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
        mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
        diff = mean_inc - mean_con

        warnings.warn(
            f"'{congruency}' has been recoded to 0 -> {output_coding_con} / 1 -> {output_coding_inc}. "
            f"RT Difference between incongruent - congruent conditions: {diff}."
        )

    elif congruency_labels == {"congruent", "incongruent"}:
        mapping = {"congruent": output_coding_con, "incongruent": output_coding_inc}
        data[congruency] = data[congruency].map(mapping).astype("object")

        mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
        mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
        diff = mean_inc - mean_con

        if {output_coding_con, output_coding_inc} != {"congruent", "incongruent"}:
            warnings.warn(
                f"'{congruency}' has been recoded to congruent -> {output_coding_con} / incongruent -> {output_coding_inc}. "
                f"RT Difference between incongruent - congruent conditions: {diff}."
            )

    # final sanity check (runs for all cases)
    mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
    mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
    diff = mean_inc - mean_con
    if diff < 0:
        warnings.warn(
            f"RT Difference between incongruent - congruent conditions is negative: {diff}. "
            f"Please check the coding of congruency conditions."
        )

    return data


def format_empirical_data(
    data: pd.DataFrame,
    rt: str = None,
    accuracy: str = None,
    congruency: str = None,
) -> Dict[str, np.ndarray]:
    """
    Formats empirical behavioral data into a structured dictionary for model inference.

    This function extracts specified variables from a pandas DataFrame, converts them 
    to a NumPy-based dictionary format, and reshapes the data to align with the expected 
    input dimensions of a probabilistic model or training pipeline.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing empirical data, typically with columns representing 
        response time ('rt'), accuracy, and experimental conditions.

    rt : str
        Column name for reaction time in empirical data set.
    accuracy : str
        Column name for accuracy in empirical data set.
    congruency : str
        Column name for congruency (coded as or 0 (congruent) /1 (incongruent)).

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - 'rt': 3D NumPy array of response times, shape (1, N, 1)
        - 'accuracy': 3D NumPy array of accuracy values, shape (1, N, 1)
        - 'conditions': 3D NumPy array of experimental condition identifiers, shape (1, N, 1)
        - 'num_obs': 2D NumPy array with the number of observations, shape (1, 1)
        
    Notes:
    ------
    The reshaping to 3D (and 2D for 'num_obs') ensures compatibility with batch-based 
    inference or training procedures where dimensions typically follow the pattern 
    (batch, number of observations, variable).
    """

    data = check_congruency(data=data, rt=rt, congruency=congruency, output_coding_con=0, output_coding_inc=1)

    var_names = [rt, accuracy, congruency]
    
    # extract relevant variables
    data_np = data[var_names].values

    # convert to dictionary
    inference_data = dict(rt=data_np[:,0],
                          accuracy=data_np[:,1],
                          conditions=data_np[:,2])

    # add dimensions so it fits training data
    inference_data = {k: v[np.newaxis,..., np.newaxis] for k, v in inference_data.items()}

    # adjust dimensions of num_obs
    inference_data["num_obs"] = np.array([data_np.shape[0]])[:,np.newaxis]
    
    return inference_data


def fit_empirical_data(
    data: pd.DataFrame,
    approximator: Any,
    num_samples: int = 1000,
    id_name: str = "id",
    rt: str = None,
    accuracy: str = None,
    congruency: str = None,
) -> pd.DataFrame:
    """
    Samples posteriors for empirical data for each unique subject or group.

    This function iterates over unique identifiers in the input DataFrame (e.g., participants),
    formats their data appropriately, performs posterior sampling using the specified 
    approximator, and aggregates the results into a combined DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing empirical observations. Must include a column corresponding
        to `id_name` to distinguish between different units (e.g., participants).
    
    approximator : bayesflow.approximators.ContinuousApproximator
        A trained BayesFlow `ContinuousApproximator` object used to perform amortized 
        posterior inference. It must implement a `.sample(conditions, num_samples)` method,
        where `conditions` is a dictionary of formatted input data and `num_samples` 
        is the number of posterior samples to draw.

    id_name : str, optional
        The column name used to identify unique units in the data (e.g., "participant").
        Defaults to "id".

    rt : str
            Column name for reaction time in empirical data set.
    accuracy : str
        Column name for accuracy in empirical data set.
    congruency : str
        Column name for congruency (coded as or 0 (congruent) /1 (incongruent)).

    Returns:
    --------
    pandas.DataFrame
        A concatenated DataFrame containing posterior samples for all individuals.
        Includes:
        - Flattened posterior samples (one column per variable)
        - The participant/group identifier (`id_name`)
        - The sampling time for each individual (`sampling_time`)

    Notes:
    ------
    - This function assumes that the `format_empirical_data` function is available
      and correctly formats individual data into a dictionary suitable for the 
      approximator.
    - The `approximator` must support a `sample` method with arguments:
      `conditions` (dict) and `num_samples` (int).
    """
    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        tqdm.write(f"{category.__name__}: {message}")

    warnings.showwarning = custom_warning_handler

    check_vars(data=data, rt=rt, accuracy=accuracy, congruency=congruency, id_name=id_name)

    # extract unique id labels
    ids=data[id_name].unique()

    list_data_samples=[]

    # iterate over participants
    for i in tqdm(range(0, len(ids)), desc="Sampling posteriors"):

        id = ids[i]
        
        # select participant data
        part_data = data[data[id_name]==id]
        
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            # --- Your function call ---
            part_data = format_empirical_data(
                part_data,
                rt=rt,
                accuracy=accuracy,
                congruency=congruency
            )

            # --- If a warning occurred ---
            for w in caught:
                tqdm.write(
                    f"[ID {id}] {w.category.__name__}: {w.message}"
                )

        # draw posterior samples with the given approximator
        start_time=time.time()
        samples = approximator.sample(conditions=part_data, num_samples=num_samples)
        end_time=time.time()
        
        # computing total sampling time
        sampling_time=end_time-start_time

        # reformat it back into a numpy array -> DataFrame
        samples_2d={k: v.flatten() for k, v in samples.items()}
        
        data_samples=pd.DataFrame(samples_2d)
        
        data_samples[id_name]=id
        data_samples["sampling_time"]=sampling_time
        
        list_data_samples.append(data_samples)

    # combine data frames from all participants
    data_samples_complete=pd.concat(list_data_samples)

    return data_samples_complete


def weighted_metric_sum(
    metrics_table: pd.DataFrame,
    weight_recovery: float = 1.0,
    weight_pc: float = 1.0,
    weight_sbc: float = 1.0,
) -> float:
    """
    Computes a weighted sum of model evaluation metrics to produce a single scalar score.

    This function takes a table of metrics (e.g., parameter recovery, posterior contraction, 
    simulation-based calibration) and computes a weighted average score that can be used 
    to compare models or configurations. The second row (posterior contraction) is transformed 
    by subtracting it from 1, assuming smaller values are better.

    Parameters:
    -----------
    metrics_table : pandas.DataFrame or numpy.ndarray
        A 2D structure where each row corresponds to a different metric and each column 
        corresponds to a parameter or evaluation dimension. The expected row order is:
        0 - Recovery
        1 - Posterior Contraction (will be inverted internally)
        2 - Simulation-Based Calibration (SBC)

    weight_recovery : float, optional
        Weight assigned to the recovery metric. Default is 1.

    weight_pc : float, optional
        Weight assigned to the posterior contraction metric. Default is 1.

    weight_sbc : float, optional
        Weight assigned to the SBC metric. Default is 1.

    Returns:
    --------
    float
        A single scalar value representing the weighted sum of the mean metrics across parameters.

    Notes:
    ------
    - Posterior contraction values are assumed to be better when smaller, so they are
      transformed using `1 - value` to reward narrower posteriors.
    - All metrics are averaged across parameters before weighting.
    - This function assumes the metrics are in the expected row order.
    """
    
    # recode posterior contraction
    mt = metrics_table.copy()
    mt.iloc[1, :] = 1 - mt.iloc[1, :]

    # compute means across parameters
    metrics_means=mt.mean(axis=1)

    # decide on weights for each metric (Recovery, Posterior Contraction, SBC)
    metrics_weights=np.array([weight_recovery, weight_pc, weight_sbc])

    # compute weighted sum
    weighted_sum=np.dot(metrics_means, metrics_weights)
    
    return weighted_sum


def post_samples_to_df(post_samples):
    """
    Convert batched posterior samples into a long-format pandas DataFrame.

    This function takes posterior samples stored as a dictionary of NumPy arrays
    (e.g., as returned by a BayesFlow approximator) and converts them into a single
    concatenated pandas DataFrame. Each batch element (e.g., participant or dataset)
    is assigned a unique integer identifier via an `id` column.

    Parameters
    ----------
    post_samples : dict
        Dictionary of posterior samples. Each key corresponds to a model parameter
        name, and each value must be a NumPy array with shape
        `(n_ids, n_samples, n_dims)`, where:
        - `n_ids` is the number of independent units (e.g., participants),
        - `n_samples` is the number of posterior samples per unit,
        - `n_dims` is the parameter dimensionality (typically 1).

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame containing posterior samples with one row per sample.
        The DataFrame includes:
        - One column per parameter in `post_samples`
        - An integer column `id` identifying the originating unit

        The total number of rows is `n_ids × n_samples`.

    Notes
    -----
    - All parameter arrays in `post_samples` are assumed to have identical shapes
      in their first two dimensions (`n_ids`, `n_samples`).
    - Parameter arrays are flattened along the last dimension before insertion
      into the DataFrame.
    - The `id` column is zero-indexed and assigned in the order of the first
      dimension of the arrays.

    Examples
    --------
    >>> df = post_samples_to_df(post_samples)
    >>> df.head()
           A   tau   mu_c   mu_r     b  id
    0  0.45  0.32   1.12   0.98  0.21   0
    """

    ids = post_samples['A'].shape[0]

    lst_samples = []

    for id in range(0, ids):

        samples_2d = {k: v[id, :, :].flatten() for k, v in post_samples.items()}
        
        df_single = pd.DataFrame(samples_2d)

        df_single['id'] = id

        lst_samples.append(df_single)

    return pd.concat(lst_samples)

def resim_data_id(
    post_sample_data: Union[pd.DataFrame, Mapping[str, np.ndarray]],
    num_obs: int,
    simulator: Any,
    id: Union[str, int],
    id_name: Union[str, int] = 'id',
    num_resims: int = 50,
    param_names: Sequence[str] = ("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
    lower_bound: float = 0
) -> pd.DataFrame:
    """
    Resimulates data based on posterior parameter samples for a given participant.

    This function takes posterior samples, filters out invalid values (e.g., negatives), and uses 
    them to generate synthetic datasets by repeatedly calling a simulator. It supports both fixed 
    and variable `sd_r` scenarios depending on the simulator configuration.

    Parameters:
    -----------
    post_sample_data : pandas.DataFrame
        A DataFrame containing posterior samples for model parameters. Each column should correspond 
        to a parameter (e.g., "A", "tau", "mu_c", etc.).

    num_obs : int
        The number of observations (e.g., trials) to simulate for each resimulation. Typically matches 
        the size of the empirical dataset.

    simulator : object
        A simulator object with an `experiment(...)` method that accepts the relevant parameters 
        and returns simulated data in a tabular format (e.g., list of dicts or DataFrame-compatible structure). 
        The object may also have an attribute `sdr_fixed` which controls whether `sd_r` is passed explicitly.

    id : str or int
        Specific id for whom the resimulations are being generated.

    num_resims : int, optional
        The number of independent resimulation runs to perform. Default is 50.

    param_names : list of str, optional
        The list of parameter names to consider when filtering and passing values to the simulator. 
        These should match the columns in `post_sample_data`. Default is ["A", "tau", "mu_c", "mu_r", "b"].

    lower_bound : float
        Values that fall below the specified value are excluded before resimulating data.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing all simulated trials across resimulations. Includes:
        - Simulated trial data from the `simulator`
        - A "num_resim" column indicating the resimulation index
        - A "id" column identifying the source participant

    Notes:
    ------
    - Posterior samples with negative values are excluded before resimulation. The number of 
      excluded samples is tracked but not returned; consider logging or returning `excluded_samples` if needed.
    - The function assumes that enough valid (non-negative) samples are available to perform `num_resims`.
    - If `simulator.sdr_fixed` is not `None`, `sd_r` will not be passed as a parameter.

    """

    # convert to dict (allow differing number of samples per parameter)

    if ~isinstance(post_sample_data, dict):
        resim_samples = dict(post_sample_data)

    # count excluded samples
    excluded_samples = dict()

    excluded_samples['num_samples'] = post_sample_data.shape[0]
    excluded_samples[id_name] = id


    # exclude negative samples
    for k, dat in resim_samples.items():
        if k in param_names:
            samples = dat.values[dat.values >= lower_bound]
            np.random.shuffle(samples)
            resim_samples[k] = samples

            excluded_samples[k] = dat.values[dat.values < 0].shape[0]

    list_resim_dfs = []

    # resimulate
    for i in range(num_resims):

        iteration_dict = {key: values[i] for key, values in resim_samples.items() if key in param_names}

        resim =  simulator.experiment(**iteration_dict | {'num_obs': num_obs})

        resim_df = pd.DataFrame(resim)
        
        resim_df["num_resim"] = i
        resim_df[id_name] = id
        
        list_resim_dfs.append(pd.DataFrame(resim_df))

    resim_complete = pd.concat(list_resim_dfs)

    return resim_complete

def resim_data(empirical_data: pd.DataFrame, 
               post_samples: pd.DataFrame,
               simulator,
               num_resims: int = 50,
               param_names: Sequence[str] = ("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
               rt: str = 'rt',
               id_name: str = 'id',
               congruency: str = 'congruency',
               simulator_congruency: str = 'conditions',
               simulator_congruency_coding: float = 0.0,
               simulator_incongruency_coding: float = 1.0,
               exclude_nonconvergents: bool = True,
               lower_bound: float = 0):
    
    """
    Perform posterior-predictive resimulations for each unit in an empirical dataset.

    This function loops over all unique identifiers in `empirical_data[id_name]`,
    determines the number of empirical observations per identifier, subsets the
    corresponding posterior parameter samples from `post_samples`, and calls
    `resim_data_id(...)` to generate resimulated trial-level data via `simulator`.

    After simulation, the function:
    1) removes non-convergent trials (defined as `rt == -1`),
    2) recodes the numeric condition codes in the `conditions` column into a
       human-readable congruency label column (`congruency`) using the mapping
       `{0.0: "congruent", 1.0: "incongruent"}`.

    Parameters
    ----------
    empirical_data : pandas.DataFrame
        Empirical trial-level dataset containing at least the identifier column
        `id_name`. The number of rows per identifier determines `num_obs` passed
        to the simulator.

    post_samples : pandas.DataFrame
        Long-format posterior samples containing at least the identifier column
        `id_name`. For each identifier, this function selects the subset
        `post_samples[post_samples[id_name] == part]` and passes it to
        `resim_data_id(...)`.

    rt: str,
        Name of the reaction time variable in `data.

    simulator : object
        A simulator instance compatible with `resim_data_id(...)` (typically
        exposing an `experiment(...)` method).

    id_name : str, optional
        Name of the identifier column used to match empirical units to posterior
        samples. Default is `'id'`.

    congruency : str, optional
        Name of the output column storing congruency labels derived from the
        numeric simulator_congruency column. Default is `'congruency'`.

    simulator_congruency : str
        Name of the congruency conditions variable as simulated by the simulator.

    simulator_congruency_coding : float
        values/ label of the congruent condition in the simulator_congruency variable. Default is `0.0`.

    simulator_incongruency_coding: float
        values/ label of the incongruent condition in the simulator_congruency variable. Default is `1.0`.

    exclude_nonconvergents: bool
        Indicates if nonconvergent trials (rt = -1) should be excluded. Default is `True`.

    Returns
    -------
    list[pandas.DataFrame]
        A list of per-identifier resimulated datasets. Each element is a
        trial-level DataFrame produced by `resim_data_id(...)`, filtered to remove
        `rt == -1` rows and augmented with a congruency label column
        (`congruency`).

    External Dependencies / Assumptions
    -----------------------------------
    - `resim_data_id(...)` must be defined in the surrounding scope and accept
      arguments compatible with:
        `resim_data_id(part_data_samples, num_obs, simulator, id, param_names=param_names)`
    - `param_names` must exist in the surrounding scope (global or closure).
    - The resimulated output is expected to contain columns:
        - `'rt'` (reaction time; used to filter non-convergents)
        - `'conditions'` (numeric condition codes; used for congruency mapping)

    Notes
    -----
    - If `post_samples` is missing entries for an identifier in `empirical_data`,
      the corresponding resimulation may be empty or raise an error inside
      `resim_data_id(...)` depending on its implementation.
    - The congruency mapping assumes exactly two condition codes: 0.0 and 1.0.
      If your simulator uses different coding, adjust the mapping accordingly.
    """

    check_vars(data=empirical_data, id_name=id_name, rt=rt, congruency=congruency)

    ids = empirical_data[id_name].unique()

    lst_data = []

    for i in tqdm(range(0, len(ids)), desc=f"Resimulate {num_resims} data sets per ID"):
        
        id = ids[i]

        num_obs = empirical_data[(empirical_data[id_name] == id)].shape[0]

        part_data_samples = post_samples[post_samples[id_name]==id]

        # resimulate data
        data_resimulated = resim_data_id(part_data_samples, num_obs=num_obs, num_resims=num_resims, simulator=simulator, id=id, param_names=param_names, lower_bound=lower_bound)
        
        # exclude non-convergents
        if exclude_nonconvergents:
            data_resimulated = data_resimulated[data_resimulated[rt] != -1]

        # recode congruency
        data_resimulated[congruency] = data_resimulated[simulator_congruency].map({simulator_congruency_coding: "congruent", simulator_incongruency_coding: "incongruent"})

        lst_data.append(data_resimulated)

    return pd.concat(lst_data)

def param_labels(param_names):
    """
    Formats a list of parameter names for LaTeX-style labeling (e.g., for plotting).

    This function wraps each parameter name in LaTeX math mode formatting, optionally adding 
    a backslash prefix (`\\`) for specific Greek-like symbols (e.g., "tau", "mu_c", "mu_r"), 
    which are typically rendered as LaTeX commands (e.g., "\\tau").

    Parameters:
    -----------
    param_names : list of str
        A list of parameter names (e.g., ["A", "tau", "mu_c"]) to be formatted.

    Returns:
    --------
    list of str or str
        A list of LaTeX-formatted strings if the input contains multiple parameters,
        or a single formatted string if only one parameter is provided.

    Examples:
    ---------
    >>> param_labels(["A", "tau", "mu_c"])
    ['$A$', '$\\tau$', '$\\mu_c$']

    >>> param_labels(["tau"])
    '$\\tau$'

    Notes:
    ------
    - The function assumes that any parameter in ["tau", "mu_c", "mu_r"] should be interpreted 
      as a LaTeX symbol and prefixed with a backslash.
    - The returned strings can be used directly as axis labels in Matplotlib or other plotting libraries
      that support LaTeX-style rendering.
    """

    param_labels = []

    for p in param_names:

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        param_labels.append(suff + p + "$")

    if len(param_labels) <= 1:
        param_labels = param_labels[0]
        
    return param_labels

def smd_samples(
    samples1: pd.DataFrame,
    samples2: pd.DataFrame,
    param_names: List[str],
    num_samples: int = 1000,
    sharex: bool = True,
    id_name: str = 'id',
    hdi_color: str = 'white',
    hdi_alpha: float = 1.0,
    x_prop: float = 0.05,
    y_prop: float = 0.85,
    zero_line: bool = True,
    x_lower: float = -1.2,
    x_upper: float = 1.2,
    fontsize: int = 15,
    fontsize_ticklabels: int = 12,
    fontsize_label: int = 15,
    fontsize_axis_labels: int = 15,
    figsize: Tuple[float, float] = (15.0, 3.0),
    supxlabel: str = 'Standardized Mean Difference $d_i$'
) -> Tuple[pd.DataFrame, Figure]:
    """
    Computes and visualizes Cohen's d for paired posterior parameter samples across multiple participants.

    This function calculates standardized mean differences (Cohen's d) between two posterior sample sets 
    (e.g., from different experimental conditions) for each parameter of interest. The differences are 
    computed across participants for each Monte Carlo sample and summarized via KDE plots, including 
    posterior means and 95% highest density intervals (HDIs).

    Parameters:
    -----------
    samples1 : pandas.DataFrame
        Posterior samples from condition 1 (e.g., control), with one column per parameter and one row 
        per sample per participant. Must include a `id_name` column.

    samples2 : pandas.DataFrame
        Posterior samples from condition 2 (e.g., experimental), formatted identically to `samples1`.

    param_names : list of str
        Names of the parameters for which Cohen's d should be computed.

    num_samples : int, optional
        Number of Monte Carlo samples to use for computing Cohen's d. Default is 1000.

    sharex : bool, optional
        Whether the x-axis should be shared across subplots. Default is True.

    id_name : str, optional
        Column name identifying the subject or participant in both sample sets. Default is 'id'.

    hdi_color : str, optional
        Color used for the KDE line. Default is 'white'.

    hdi_alpha : float, optional
        Alpha transparency level for the filled KDE. Default is 1 (opaque).

    x_prop : float, optional
        Proportional x-position (in axis coordinates) for placing the mean d text label. Default is 0.05.

    y_prop : float, optional
        Proportional y-position (in axis coordinates) for placing the mean d text label. Default is 0.85.

    zero_line : bool, optional
        Whether to draw a vertical line at d = 0 for visual reference. Default is True.

    x_lower : float, optional
        Lower bound of the x-axis for all subplots. Default is -1.2.

    x_upper : float, optional
        Upper bound of the x-axis for all subplots. Default is 1.2.

    Returns:
    --------
    data_d : pandas.DataFrame
        DataFrame containing Cohen's d values across all Monte Carlo samples for each parameter.

    fig : matplotlib.figure.Figure
        Figure containing the KDE plots for each parameter's standardized mean difference distribution.

    Notes:
    ------
    - Assumes the same number of participants and sample structure in both `samples1` and `samples2`.
    - Issues warnings if participant IDs do not match between samples.
    - Uses standard deviation of paired differences as the denominator for computing Cohen's d.
    - Uses seaborn for density visualization and matplotlib for figure layout.
    - This function is intended for paired comparison designs where within-subject parameter estimates are compared.

    Example:
    --------
    >>> smd_samples(samples_control, samples_treatment, ["A", "tau", "mu_c"])
    """

    num_params = len(param_names)
    cohens_ds = np.ones((num_samples,num_params))

    parts = samples1[id_name].unique()

    # deterministic draw index within each participant
    samples1 = samples1.copy()
    samples2 = samples2.copy()

    samples1["sample_id"] = samples1.groupby(id_name).cumcount()
    samples2["sample_id"] = samples2.groupby(id_name).cumcount()

    # choose the same draw indices for both conditions
    draws = np.random.choice(num_samples, size=num_samples, replace=False)

    samples1 = samples1[samples1["sample_id"].isin(draws)]
    samples2 = samples2[samples2["sample_id"].isin(draws)]


    for j,p in enumerate(param_names):
        for i in range(0, num_samples):
            # control condition
            m1 = samples1[samples1['sample_id'] == i][p]
            #m1 = m1[~np.isnan(m1)]

            # experimental manipulation
            m2 = samples2[samples2['sample_id'] == i][p]
            #m2 = m2[~np.isnan(m2)]

            if set(samples1[samples1['sample_id'] == i][id_name].unique()) != set(parts):
                warnings.warn(f'Participants in sub sample 1 and sample id {i} are not identical to all participants!')
            
            if set(samples2[samples2['sample_id'] == i][id_name].unique()) != set(parts):
                warnings.warn(f'Participants in sub sample 2 and sample id {i} are not identical to all participants!')

            if m1.shape[0] != parts.shape[0] or m2.shape[0] != parts.shape[0]:
                warnings.warn(f'Mismatch in number of entries in sample id {i}')

            m1 = m1.values
            m2 = m2.values

            d = np.mean(m1) - np.mean(m2)

            diff = m1 - m2
            sd = np.std(diff, ddof=1)
            mean_d = np.nan if sd == 0 else d / sd

            cohens_ds[i,j] = mean_d

    data_d = pd.DataFrame(cohens_ds, columns = param_names)

    
    fig, axes = plt.subplots(1, len(param_names), figsize=figsize, sharex=sharex)

    for p, ax in zip(param_names, axes):

        ax.set_xlim(x_lower, x_upper)

        post_mean = np.mean(data_d[p])
        ax.axvline(x=post_mean, color='black', linestyle='--', linewidth=1)

        if zero_line:
            ax.axvline(x=0, color='red', linestyle='-', linewidth=1)

        #ax.set_xlim(x_lower, x_upper)
        hdi_bounds = hdi(data_d[p].values, hdi_prob=0.95)

        # HDI as shaded region with a different, subtle color
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=True, alpha=0.3,linewidth=0)
        ax.axvspan(ax.get_xlim()[0], hdi_bounds[0], color=hdi_color, alpha= hdi_alpha)  # Left of HDI
        ax.axvspan(hdi_bounds[1], ax.get_xlim()[1], color=hdi_color, alpha= hdi_alpha)  # Right of HDI
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=False, alpha=1,linewidth=1)

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        label = suff + p + "$"

        ax.set_title(label, fontsize=fontsize)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=fontsize_ticklabels)  


        if p == 'A':
            ax.set_ylabel('Density', fontsize=fontsize_axis_labels)
        else:
            ax.set_ylabel('')

        ymax = ax.get_ylim()[1]
        xmin = ax.get_xlim()[0]
        xmax = ax.get_xlim()[1]

        x_range = xmax-xmin

        ax.text(xmin + x_range*x_prop, ymax*y_prop, '$d = $' + str(round(post_mean, 2)), fontsize=fontsize_label, color='black', rotation=0)
    
    fig.supxlabel(supxlabel, fontsize=fontsize)
    fig.tight_layout()

    return data_d, fig



def format_sim_data(
    sim_data: Dict[str, np.ndarray],
    congruency_coding: int = 0,
    only_convergents: bool = True,
    id_name: str = 'id'
) -> pd.DataFrame:
    """
    Format simulated behavioral data into a long-format pandas DataFrame.

    This function takes batched simulation output (reaction times, accuracy,
    and condition codes) and converts it into a single concatenated DataFrame
    suitable for downstream statistical analysis or visualization (e.g. compute_stats, plot_stats).

    Parameters
    ----------
    sim_data : Dict[str, np.ndarray]
        Dictionary containing simulation outputs with the following keys:
        - 'rt': Reaction times, shape (batch_size, n_trials, 1)
        - 'accuracy': Accuracy values, shape (batch_size, n_trials, 1)
        - 'conditions': Condition codes, shape (batch_size, n_trials, 1)
    congruency_coding : int, optional
        Integer code indicating a congruent condition in `conditions`.
        All other values are treated as incongruent. Default is 0.
    only_convergents : bool, optional
        If True, remove trials with reaction time equal to -1,
        which are assumed to represent non-convergent simulations.
        Default is True.
    id_name : str, optional
        Variable name for data set identifier. Default to 'id',

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per trial and the following columns:
        - 'rt': Reaction time
        - 'accuracy': Accuracy value
        - 'conditions': Condition code
        - 'id': Batch index
        - 'congruency': 'congruent' or 'incongruent'
        - 'accuracy_name': 'correct' or 'incorrect'
    """
    batch_size: int = sim_data['rt'].shape[0]

    behav_keys = ['rt', 'accuracy', 'conditions']
    behav_data: Dict[str, np.ndarray] = {k: sim_data[k] for k in behav_keys}

    df_list = []
    rt_var = 'rt'

    for i in range(batch_size):
        stacked = np.stack(
            (
                behav_data['rt'][i, :, :],
                behav_data['accuracy'][i, :, :],
                behav_data['conditions'][i, :, :]
            ),
            axis=1
        )[:, :, 0]

        df_single = pd.DataFrame(stacked, columns=[rt_var, 'accuracy', 'conditions'])
        df_single[id_name] = i

        df_single['congruency'] = [
            'congruent' if x == congruency_coding else 'incongruent'
            for x in df_single['conditions']
        ]
        df_single['accuracy_name'] = [
            'correct' if x == 1.0 else 'incorrect'
            for x in df_single['accuracy']
        ]

        df_list.append(df_single)

    df_complete = pd.concat(df_list, ignore_index=True)

    if only_convergents:
        df_complete = df_complete[df_complete[rt_var] != -1]

    return df_complete


def compute_stats(
    data: pd.DataFrame,
    id_name: str = "id",
    n_rt_bins: int = 5,
    rt: str = 'rt',
    accuracy: str = 'accuracy',
    congruency: str = "congruency",
    quantiles: Union[np.ndarray, Sequence[float]] = np.arange(0.1, 1.0, 0.1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute distributional summary statistics for reaction-time (RT) data, producing
    inputs suitable for CAF, CDF, and Δ-function plots.

    This function derives three DataFrames:

    1. **Δ-function data (`delta_data`)**:
       Quantiles of RT computed *only on correct trials* (``accuracy == 1``) for each
       ``id_name`` × ``congruency`` group, then pivoted to wide format with
       separate columns per congruency level (expected: ``'congruent'`` and
       ``'incongruent'``). It additionally computes:

       - ``delta = incongruent - congruent``
       - ``mean_qu = (incongruent + congruent) / 2``

    2. **CAF data (`caf_data`)**:
       Mean accuracy per RT bin (quantile bins over ``rt``), stratified by
       ``id_name`` × ``congruency`` × ``rt_bin``.

    3. **CDF data (`cdf_data`)**:
       Long-format representation of the wide quantile RTs from `delta_data`, with
       columns ``[id_name, quantile, condition, rt]`` suitable for CDF plotting.

    Parameters
    ----------
    data : pandas.DataFrame
        Trial-level (long-format) data containing RTs and accuracy. Required columns:

        - ``'rt'`` : float
            Reaction time (typically seconds).
        - ``'accuracy'`` : int | bool | float
            Trial accuracy indicator. Trials with ``accuracy == 1`` are treated as
            correct for Δ-function quantiles.
        - ``{id_name}`` : hashable (e.g., int | str)
            Identifier for subject/session/batch.
        - ``{congruency}`` : str-like / categorical
            Congruency label. The Δ-function computation assumes that the pivot will
            yield columns named ``'congruent'`` and ``'incongruent'``.

        Notes
        -----
        The function adds/overwrites a column ``'rt_bin'`` in ``df_complete`` (in-place)
        computed via ``pandas.qcut``.

    id_name : str, default='id'
        Column name identifying independent units (e.g., participant, session, batch).

    congruency : str, default='congruency'
        Column name indicating congruency condition. For downstream computations,
        the values are expected to include levels that pivot to columns named
        ``'congruent'`` and ``'incongruent'``.

    n_rt_bins : int, default=5
        Number of quantile bins used to discretize RTs for the CAF computation.
        Implemented with ``pandas.qcut`` (approximately equal-sized bins).

    rt : str, default='rt'
        Variable name of the reaction time variable.
    
    accuracy : str, default='accuracy'
        Variable name of the accuracy (1 = correct response, 0 = incorrect response)

    quantiles : numpy.ndarray or Sequence[float], default=np.arange(0.1, 1.0, 0.1)
        Quantile levels at which to compute RT quantiles for correct trials. Values
        should lie in the open interval (0, 1].

    Returns
    -------
    caf_data : pandas.DataFrame
        DataFrame containing conditional accuracy values per RT bin. Expected columns:

        - ``{id_name}``
        - ``{congruency}``
        - ``'rt_bin'`` : int
        - ``'accuracy'`` : float

    cdf_data : pandas.DataFrame
        Long-format CDF-ready DataFrame with columns:

        - ``{id_name}``
        - ``'quantile'`` : float
        - ``'condition'`` : str
        - ``'rt'`` : float

    delta_data : pandas.DataFrame
        Wide-format DataFrame with per-``id_name`` quantiles for each congruency level,
        plus derived columns ``delta`` and ``mean_qu``. Expected columns include:

        - ``{id_name}``
        - ``'quantile'`` : float
        - ``'congruent'`` : float
        - ``'incongruent'`` : float
        - ``'delta'`` : float
        - ``'mean_qu'`` : float

    Raises
    ------
    KeyError
        If required columns are missing from ``data``.
    ValueError
        If ``pandas.qcut`` fails (e.g., due to too many duplicate RT values causing
        non-unique bin edges), or if the required congruency levels do not produce
        ``'congruent'`` and ``'incongruent'`` columns after pivoting.

    Examples
    --------
    >>> caf_data, cdf_data, delta_data = compute_stats(data, id_name="subject_id")
    >>> # Pass outputs to plotting utilities
    >>> fig, axes = plot_stats(caf_data, cdf_data, delta_data, id_name="subject_id")
    """

    check_vars(data=data, rt=rt, accuracy=accuracy, congruency=congruency, id_name=id_name)

    data = check_congruency(data=data, rt=rt, congruency=congruency, output_coding_con='congruent', output_coding_inc='incongruent')

    data[rt] = pd.to_numeric(data[rt], errors="coerce")

    delta_data = (
        data[data[accuracy] == 1]
        .groupby([id_name, congruency])[rt]
        .quantile(quantiles)
        .reset_index()
        .rename(columns={"level_2": "quantile"})
        .pivot(index=[id_name, "quantile"], columns=[congruency], values=rt)
        .reset_index()
        .assign(delta=lambda df: df["incongruent"] - df["congruent"])
        .assign(mean_qu=lambda df: (df["incongruent"] + df["congruent"]) / 2)
    )

    df = data.copy()

    df["rt_bin"] = pd.qcut(df[rt], q=n_rt_bins, labels=False)

    caf_data = (
        df.groupby([id_name, congruency, "rt_bin"])[accuracy]
        .mean()
        .reset_index()
        .rename(columns={"level_2": "quantile"})
        .reset_index()
    )

    cdf_data = pd.melt(
        delta_data,
        id_vars=[id_name, "quantile"],
        value_vars=["congruent", "incongruent"],
        var_name=congruency,
        value_name=rt,
    )

    return caf_data, cdf_data, delta_data


def plot_stats(
    caf_data: pd.DataFrame,
    cdf_data: pd.DataFrame,
    delta_data: pd.DataFrame,
    alpha: float = 0.05,
    id_name: str = "id",
    congruency: str = "congruency",
    rt : str = 'rt',
    n_delta_bins: int = 10,
    fontsize: int = 24,
    fontsize_axes: int = 15,
    delta_ylim: Optional[Tuple[float, float]] = None,
    delta_xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """
    Plot three standard distributional diagnostics for reaction-time (RT) data:
    (1) conditional accuracy function (CAF), (2) cumulative distribution function (CDF),
    and (3) a delta-function summary of condition differences across the RT distribution.

    The function creates a single figure with three subplots arranged horizontally:

    1. **CAF**: Accuracy as a function of binned RT (``rt_bin``), stratified by
       ``congruency``.
    2. **CDF**: Empirical CDFs (quantile vs. RT) for each ``condition``. Individual
       trajectories are shown per ``id_name`` (faint lines) and an overlaid mean CDF
       is shown per ``condition``.
    3. **Δ-function**: Condition difference (``delta``) as a function of mean RT quantile
       (``mean_qu``). Individual trajectories are shown per ``id_name`` (very faint)
       with an aggregated (mean-by-quantile) curve overlaid.

    Parameters
    ----------
    delta_data : pandas.DataFrame
        Long-format data required for the Δ-function panel. Must contain at least:

        - ``'quantile'``: Quantile index/label (used for aggregation).
        - ``'mean_qu'``: Mean RT associated with each quantile (x-axis of Δ-function).
        - ``'delta'``: Difference metric to plot (y-axis of Δ-function).
        - A column named by ``id_name``: Identifier for individual trajectories.

        Notes
        -----
        The function will add a temporary column ``'mean_qu_bins'`` via ``pd.cut``.
        (It is overwritten if already present.)

    caf_data : pandas.DataFrame
        Data for the CAF panel. Must codf_longntain at least:

        - ``'rt_bin'``: RT bin index/label (x-axis of CAF).
        - ``'accuracy'``: Accuracy per bin (y-axis of CAF).
        - A column named by ``congruency``: Grouping variable for CAF lines.

    cdf_data : pandas.DataFrame
        Long-format data for the CDF panel. Must contain at least:

        - ``'rt'``: Reaction times in seconds (x-axis of CDF).
        - ``'quantile'``: CDF quantiles (y-axis of CDF).
        - ``'condition'``: Condition label for grouping/colouring.
        - A column named by ``id_name``: Identifier for individual trajectories.

    alpha : float, default=0.05
        Opacity for individual CDF trajectories (panel 2). The mean CDF is plotted with
        opacity 1.0.

    id_name : str, default='id'
        Column name used as an identifier for individual trajectories in the CDF and
        Δ-function panels.

    congruency : str, default='congruency'
        Column name used to stratify the CAF panel.

    rt: str, default='rt'
        Columns name of the reaction time variable.

    n_delta_bins : int, default=10
        Number of bins used when discretizing ``delta_data['mean_qu']`` into
        ``'mean_qu_bins'``. (The function currently computes a binned summary, but then
        replaces it with a mean-by-quantile aggregation for plotting.)

    fontsize : int, default=24
        Font size for subplot titles.

    fontsize_axes : int, default=20
        Font size for axis labels.

    delta_ylim : tuple[float, float] | None, default=None
        If provided (truthy), apply a fixed y-axis range to the Δ-function panel.

    delta_xlim : tuple[float, float] | None, default=None
        If provided (truthy), apply a fixed x-axis range to the Δ-function panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.

    axes : numpy.ndarray of matplotlib.axes.Axes
        Array of axes in the order ``[CAF, CDF, Δ-function]``.

    Notes
    -----
    - This function assumes that ``matplotlib.pyplot`` is imported as ``plt``,
      ``seaborn`` as ``sns``, and ``pandas`` as ``pd`` in the calling scope.
    - The Δ-function panel uses very low opacity (``alpha=0.05``) for individual
      trajectories to emphasize the aggregated curve.

    Examples
    --------
    >>> fig, axes = plot_stats(caf_data, cdf_data, delta_data, id_name="subject")
    """
    mean_data = cdf_data.groupby(["quantile", congruency])[rt].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # CAF
    sns.lineplot(caf_data, x="rt_bin", y="accuracy", hue=congruency, ax=axes[0])

    axes[0].set_title("CAF", fontsize=fontsize)
    axes[0].set_ylabel("CAF", fontsize=fontsize_axes)
    axes[0].set_xlabel("Bins", fontsize=fontsize_axes)
    axes[0].legend(title="", loc="lower right")

    # single CDF
    sns.lineplot(
        cdf_data,
        x=rt,
        y="quantile",
        hue=congruency,
        style=id_name,
        legend=False,
        ax=axes[1],
        alpha=alpha,
    )
    # mean CDF
    sns.lineplot(mean_data, x=rt, y="quantile", hue=congruency, alpha=1, ax=axes[1])

    axes[1].set_title("CDF", fontsize=fontsize)
    axes[1].set_xlabel("RT[s]", fontsize=fontsize_axes)
    axes[1].set_ylabel('Cumulative Density', fontsize=fontsize_axes)
    axes[1].get_legend().remove()

    delta_data["mean_qu_bins"] = pd.cut(delta_data["mean_qu"], bins=n_delta_bins)
    delta_bins = delta_data.groupby("mean_qu_bins", observed=False)["delta"].mean().reset_index()
    delta_bins["bin_mid"] = delta_bins["mean_qu_bins"].apply(lambda x: x.mid)

    delta_bins = (
        delta_data.groupby("quantile")[["mean_qu", "delta"]]
        .mean()
        .reset_index()
        .sort_values("mean_qu")
    )

    # single Deltas
    sns.lineplot(
        delta_data,
        linewidth=0.5,
        linestyle="--",
        marker="o",
        x="mean_qu",
        y="delta",
        hue=id_name,
        legend=False,
        ax=axes[2],
        alpha=alpha,
    )

    # aggregated Deltas
    sns.lineplot(
        delta_bins,
        linewidth=0.5,
        linestyle="--",
        marker="o",
        x="mean_qu",
        y="delta",
        legend=False,
        ax=axes[2],
        color="black",
    )

    axes[2].set_ylabel("$\\Delta$", fontsize=fontsize_axes)
    axes[2].set_xlabel("RT[s]", fontsize=fontsize_axes)
    axes[2].set_title("$\\Delta$-Function", fontsize=fontsize)

    if delta_ylim is not None:
        axes[2].set(ylim=delta_ylim)
    if delta_xlim is not None:
        axes[2].set(xlim=delta_xlim)

    fig.tight_layout()

    return fig, axes



def plot_fit(
    caf_data: pd.DataFrame,
    cdf_data: pd.DataFrame,
    delta_data: pd.DataFrame,
    caf_data_emp: pd.DataFrame,
    cdf_data_emp: pd.DataFrame,
    delta_data_emp: pd.DataFrame,
    congruency: str = "congruency",
    congruency_emp: str = "congruency",
    n_delta_bins: int = 10,
    delta_ylim: Optional[Tuple[float, float]] = None,
    delta_xlim: Optional[Tuple[float, float]] = None,
    cdf_xlim: Optional[Tuple[float, float]] = None,
    fontsize: int = 14,
    fontsize_axes: int = 14,
    fontsize_ticklabels: int = 10,
    fontsize_legend: int = 12,
    legend: bool = True,
    new_plot: bool = True,
    caf_errorbars: Optional[object] = None,
    hue_order: Sequence[str] = ("congruent", "incongruent"),
    palette_emp: Mapping[str, str] = {"congruent": "#132a70", "incongruent": "#FF6361"},
    palette_model: Mapping[str, str] = {"congruent": "#132a70", "incongruent": "#FF6361"},
    delta_linestyle_model: str = "-",
    caf_linestyle_model: str = "-",
    cdf_linestyle_model: str = "-",
    linewidth: float = 0.5,
    fig: Optional[Figure] = None,
    axes: Optional[Sequence[Axes]] = None):
    """
    Plot model and empirical CAFs, CDFs, and Δ-function in a 1×3 subplot layout.

    This function creates three panels:
    1. CAF (Conditional Accuracy Function)
    2. CDF (Cumulative Distribution Function of RTs)
    3. Δ-function (delta between conditions as a function of RT)

    Parameters
    ----------
    delta_data : pandas.DataFrame
        Model delta data with at least the columns:
        ['quantile', 'mean_qu', 'delta'].
    delta_data_emp : pandas.DataFrame
        Empirical delta data with at least the columns:
        ['quantile', 'mean_qu', 'delta'].
    caf_data : pandas.DataFrame
        Model CAF data with columns including:
        ['rt_bin', 'accuracy', <congruency>].
    caf_data_emp : pandas.DataFrame
        Empirical CAF data with columns including:
        ['rt_bin', 'accuracy', <congruency_emp>].
    cdf_data : pandas.DataFrame
        Long-format model RT data with columns:
        ['quantile', 'condition', 'rt'].
    cdf_data_emp : pandas.DataFrame
        Long-format empirical RT data with columns:
        ['quantile', 'condition', 'rt'].
    congruency : str, optional
        Column name in `caf_data` indicating congruency condition
        for the model (default: 'congruency').
    congruency_emp : str, optional
        Column name in `caf_data_emp` indicating congruency condition
        for the empirical data (default: 'congruency').
    n_delta_bins : int, optional
        Number of bins used when discretizing `mean_qu` with `pd.cut`
        (default: 10). Currently used when computing intermediate
        delta summaries.
    delta_ylim : tuple of float, optional
        Y-axis limits for the Δ-function subplot.
    delta_xlim : tuple of float, optional
        X-axis limits for the Δ-function subplot.
    cdf_xlim : tuple of float, optional
        X-axis limits for the CDF-function subplot.
    fontsize : int, optional
        Font size for subplot titles (default: 14).
    fontsize_axes : int, optional
        Font size for axis labels (default: 14).
    fontsize_ticklabels : int, optional
        Font size for tick labels (default: 10).
    fontsize_legend : int, optional
        Font size for the legend (default: 12).
    legend : bool, optional
        If True, draw a legend for the CAF panel (default: True).
    new_plot : bool, optional
        If True, create a new figure and axes. If False, draw into the
        provided `fig` and `axes` (default: True).
    caf_errorbars : object, optional
        Errorbar specification passed through to `sns.lineplot` for the model CAF.
        This can be any format accepted by seaborn's `errorbar` parameter
        (e.g. 'ci', 'se', None, a tuple, or a callable; default: None).
    hue_order : sequence of str, optional
        Order of condition levels for hue mapping (default: ('congruent', 'incongruent')).
    palette_emp : Mapping[str, str], optional
        Color palette for empirical lines, mapping condition names to hex colors
        (default: {"congruent": "#132a70", "incongruent": "#FF6361"}).
    palette_model : Mapping[str, str], optional
        Color palette for model lines, mapping condition names to hex colors
        (default: {"congruent": "#132a70", "incongruent": "#FF6361"}).
    delta_linestyle_model : str, optional
        Matplotlib linestyle for the model Δ-function line (default: '-').
    caf_linestyle_model : str, optional
        Matplotlib linestyle for the model CAF line (default: '-').
    cdf_linestyle_model : str, optional
        Matplotlib linestyle for the model CDF line (default: '-').
    linewidth : float, optional
        Line width for all plotted lines (default: 0.5).
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into when `new_plot` is False.
        Ignored if `new_plot` is True.
    axes : sequence of matplotlib.axes.Axes, optional
        Existing axes (length 3) to draw into when `new_plot` is False.
        Ignored if `new_plot` is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the 1×3 subplots.
    axes : sequence of matplotlib.axes.Axes
        The three axes objects for CAF, CDF, and Δ-function, respectively.
    """

    mean_data = cdf_data.groupby(['quantile', congruency])['rt'].mean().reset_index()

    mean_data_emp = cdf_data_emp.groupby(['quantile', congruency_emp])['rt'].mean().reset_index()

    if new_plot:
        fig, axes = plt.subplots(1,3, figsize=(12,3))

    # CAFs
    sns.lineplot(caf_data, 
                 linewidth=linewidth,
                 x='rt_bin', 
                 y='accuracy', 
                 hue=congruency, 
                 errorbar=caf_errorbars, 
                 ax=axes[0],
                 legend=False, 
                 hue_order=hue_order, 
                 palette=palette_model,
                 linestyle=caf_linestyle_model)
    
    
    sns.lineplot(caf_data_emp, 
                 linestyle='--',
                 marker="o", 
                 errorbar=None, 
                 legend=legend, 
                 linewidth=linewidth,
                 x='rt_bin', 
                 y='accuracy', 
                 hue=congruency_emp, 
                 ax=axes[0], 
                 hue_order=hue_order, 
                 palette=palette_emp)
    
    axes[0].set(ylim=(0, 1))
    axes[0].set_title('CAF', fontsize=fontsize)
    
    axes[1].set_title('CDF', fontsize=fontsize)
    axes[1].set_ylabel('Cumulative Density', fontsize=fontsize_axes)
    axes[1].set_xlabel('RT[s]', fontsize=fontsize_axes)
    axes[0].set_ylabel('CAF', fontsize=fontsize_axes)
    axes[0].set_xlabel('Bins', fontsize=fontsize_axes)

    # CDFs
    sns.lineplot(mean_data, 
                 linewidth=linewidth, 
                 linestyle=cdf_linestyle_model, 
                 x='rt', 
                 y='quantile', 
                 hue=congruency, 
                 alpha=1, 
                 ax=axes[1], 
                 legend=False, 
                 hue_order=hue_order, 
                 palette=palette_model)
    
    sns.lineplot(mean_data_emp, 
                 linewidth=linewidth, 
                 marker="o", 
                 linestyle='--', 
                 x='rt', 
                 y='quantile',
                 legend=False, 
                 hue=congruency_emp, 
                 alpha=1, 
                 ax=axes[1], 
                 hue_order=hue_order, 
                 palette=palette_emp)
    
    axes[1].set_title('CDF', fontsize=fontsize)
    axes[1].set_ylabel('Cumulative Density', fontsize=fontsize_axes)
    axes[1].set_xlabel('RT[s]', fontsize=fontsize_axes)

    # DELTA
    delta_data['mean_qu_bins'] = pd.cut(delta_data["mean_qu"], bins=n_delta_bins)
    delta_bins = delta_data.groupby('mean_qu_bins', observed=False)['delta'].mean().reset_index()
    delta_bins['bin_mid'] = delta_bins['mean_qu_bins'].apply(lambda x: x.mid)


    delta_bins = (
            delta_data
            .groupby('quantile', observed=False)[['mean_qu', 'delta']]
            .mean()
            .reset_index()
            .sort_values('mean_qu')
        )

    delta_data_emp['mean_qu_bins'] = pd.cut(delta_data_emp["mean_qu"], bins=n_delta_bins)
    delta_bins_emp = delta_data_emp.groupby('mean_qu_bins', observed=False)['delta'].mean().reset_index()
    delta_bins_emp['bin_mid'] = delta_bins_emp['mean_qu_bins'].apply(lambda x: x.mid)

    delta_bins_emp = (
            delta_data_emp
            .groupby('quantile', observed=False)[['mean_qu', 'delta']]
                .agg(
                    mean_qu=('mean_qu', 'mean'),
                    delta=('delta', 'mean'),
                    sd_delta=('delta', 'std')
                    )
            .reset_index()
            .sort_values('mean_qu')
        )

    sns.lineplot(delta_bins,linewidth=linewidth, linestyle=delta_linestyle_model, x='mean_qu', y='delta', legend=False, ax=axes[2], color='black')
    sns.lineplot(delta_bins_emp,linewidth=linewidth,linestyle='--',marker="o",  x='mean_qu', y='delta', legend=False, ax=axes[2], color='black')
    
    axes[2].set_ylabel('$\Delta$', fontsize=fontsize_axes)
    axes[2].set_xlabel('RT[s]', fontsize=fontsize_axes)
    axes[2].set_title('$\Delta$-Function', fontsize=fontsize)
    
    if cdf_xlim is not None:
        axes[2].set(xlim=cdf_xlim)

    if delta_ylim is not None:
        axes[2].set(ylim=delta_ylim)

    if delta_xlim is not None:
        axes[2].set(ylim=delta_ylim)

    if legend:
        axes[0].legend(title='', loc='lower right', fontsize=fontsize_legend, frameon=False)

    for ax in axes:
        ax.tick_params(axis='x', labelsize=fontsize_ticklabels)  
        ax.tick_params(axis='y', labelsize=fontsize_ticklabels)  

    fig.tight_layout()

    return fig, axes
   

def summarise_q(
    data: pd.DataFrame,
    rt: str = "rt",
    accuracy: str = "accuracy",
    congruency: str = "congruency",
    grouping_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute quantile-based and mean summary statistics of response times (RT)
    and accuracy within groups.

    This function aggregates trial-level data into distributional summaries
    commonly used for model fit evaluation (e.g., quantile-based fit of RT
    distributions). For each group defined by `grouping_vars`, the function
    computes:

        - RT quantiles (25th, 50th, 75th percentiles)
        - Mean RT
        - Mean accuracy

    The resulting DataFrame is returned in wide format, with one row per group
    and separate columns for each RT quantile and mean statistic.

    Parameters
    ----------
    data
        Trial-level DataFrame containing at minimum RT, accuracy, and
        congruency columns.
    rt
        Column name containing response times (numeric).
    accuracy
        Column name containing response accuracy (numeric; typically 0/1).
    congruency
        Column name indicating congruency condition labels.
    grouping_vars
        Columns used to define grouping structure (e.g., participant and
        condition). If None, defaults to ["participant", "congruency"].

    Returns
    -------
    pandas.DataFrame
        A wide-format DataFrame with one row per group defined by
        `grouping_vars`. Contains:

            - Grouping variables
            - mean_rt : mean response time per group
            - mean_acc : mean accuracy per group
            - rt_q25 : 25th percentile of RT
            - rt_q50 : 50th percentile (median) of RT
            - rt_q75 : 75th percentile of RT

    Notes
    -----
    - `check_vars()` is used to validate the presence and format of required
      columns.
    - `check_congruency()` standardizes congruency coding before aggregation.
    - RT quantiles are computed across all trials within each group; if
      quantiles should be restricted (e.g., correct trials only), filtering
      must be applied prior to calling this function.
    """
    if grouping_vars is None:
        grouping_vars = ["participant", "congruency"]

    check_vars(data, rt=rt, accuracy=accuracy, congruency=congruency)

    data = check_congruency(
        data,
        rt=rt,
        congruency=congruency,
        output_coding_con="congruent",
        output_coding_inc="incongruent",
    )

    df_q = (
        data
        .groupby(grouping_vars)[rt]
        .quantile([0.25, 0.5, 0.75])
        .rename_axis(index=[*grouping_vars, "quantile"])
        .reset_index()
    )

    df_means = (
        data
        .groupby(grouping_vars)
        .agg(
            mean_rt=(rt, "mean"),
            mean_acc=(accuracy, "mean"),
        )
        .reset_index()
    )

    df_q = df_q.merge(df_means, on=grouping_vars)

    df_q_wide = (
        df_q
        .pivot_table(
            index=grouping_vars + ["mean_rt", "mean_acc"],
            columns="quantile",
            values=rt,
        )
        .reset_index()
    )

    df_q_wide = df_q_wide.rename(
        columns={
            0.25: "rt_q25",
            0.50: "rt_q50",
            0.75: "rt_q75",
        }
    )

    return df_q_wide


def compute_fit_qs(
    resimulated_data: pd.DataFrame,
    empirical_data: pd.DataFrame,
    rt: str = "rt",
    accuracy: str = "accuracy",
    congruency: str = "congruency",
    grouping_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute and merge quantile-based summary statistics for empirical and
    resimulated datasets.

    This function summarises response time (RT) distributions and accuracy
    within groups for both empirical and resimulated data using `summarise_q()`,
    and merges the resulting summaries into a single comparison table. For each
    group defined by `grouping_vars`, the following statistics are computed:

        - RT quantiles (25th, 50th, 75th percentiles)
        - Mean RT
        - Mean accuracy

    Parameters
    ----------
    resimulated_data
        DataFrame containing model-generated (resimulated) trial-level data.
    empirical_data
        DataFrame containing empirical trial-level data.
    rt
        Column name containing reaction times.
    accuracy
        Column name containing response accuracy (expected numeric, e.g., 0/1).
    congruency
        Column name containing congruency condition labels.
    grouping_vars
        Columns used to define grouping structure (e.g., participant and
        condition). If None, defaults to ["participant", "congruency"].

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame containing quantile and mean summaries for both
        resimulated and empirical data. Columns from the two datasets are
        distinguished by suffixes:

            - '_resim' for resimulated data
            - '_emp' for empirical data

        The DataFrame is keyed by `grouping_vars`.

    Notes
    -----
    - This function does not compute formal fit statistics (e.g., correlations
      or error metrics). It prepares distributional summaries that can be used
      for subsequent model fit evaluation.
    - Internally relies on `summarise_q()` for quantile and mean computation.
    """
    if grouping_vars is None:
        grouping_vars = ["participant", "congruency"]

    df_q_emp_wide = summarise_q(
        empirical_data,
        rt=rt,
        accuracy=accuracy,
        congruency=congruency,
        grouping_vars=grouping_vars,
    )

    df_q_wide = summarise_q(
        resimulated_data,
        rt=rt,
        accuracy=accuracy,
        congruency=congruency,
        grouping_vars=grouping_vars,
    )

    data_merged = pd.merge(
        df_q_wide,
        df_q_emp_wide,
        how="left",
        on=grouping_vars,
        suffixes=("_resim", "_emp"),
    )

    return data_merged

def plot_fit_qs(
    data: pd.DataFrame,
    con_color: str = "#10225e",
    inc_color: str = "#FF6361",
    fontsize: int = 22
) -> Tuple[Figure, list[Axes]]:
    """
    Visualize quantile-based model fit by comparing empirical and resimulated
    summary statistics.

    This function generates a panel of scatterplots comparing empirical and
    resimulated summary measures (typically produced by `compute_fit_qs()`).
    For each statistic, empirical values are plotted on the x-axis and the
    corresponding resimulated values on the y-axis. A dashed diagonal
    reference line (y = x) indicates perfect agreement.

    Five statistics are visualized:

        - Mean RT
        - Mean accuracy
        - 25th percentile RT
        - Median RT (50th percentile)
        - 75th percentile RT

    Points are colored by congruency condition (expected labels:
    "congruent" and "incongruent").

    Parameters
    ----------
    data
        DataFrame containing merged empirical and resimulated summaries.
        Required columns:

            - 'congruency'
            - 'mean_rt_emp',  'mean_rt_resim'
            - 'mean_acc_emp', 'mean_acc_resim'
            - 'rt_q25_emp',   'rt_q25_resim'
            - 'rt_q50_emp',   'rt_q50_resim'
            - 'rt_q75_emp',   'rt_q75_resim'

    con_color
        Hex color used for the "congruent" condition.
    inc_color
        Hex color used for the "incongruent" condition.
    fontsize
        Font size of the figure text.


    Returns
    -------
    fig
        The Matplotlib figure.
    axes
        A list of the five subplot axes (left-to-right).

    Notes
    -----
    - Accuracy plots use fixed limits (0.6–1.0); RT plots use limits derived
      from the empirical values.
    - If your congruency labels differ, adjust `hue_order` and `palette`.
    """
    hue_order = ["congruent", "incongruent"]
    palette = {"congruent": con_color, "incongruent": inc_color}

    titles = ["Mean RT", "Mean Accuracy", "25% Quantile RT", "Median RT", "75% Quantile RT"]
    stats = ["mean_rt", "mean_acc", "rt_q25", "rt_q50", "rt_q75"]

    plot_data = data.copy()

    fig, axes_arr = plt.subplots(1, 5, figsize=(15, 3))
    axes = list(axes_arr)  # ensure a stable return type

    for j, var in enumerate(stats):
        x_col = f"{var}_emp"
        y_col = f"{var}_resim"

        sns.scatterplot(
            data=plot_data,
            x=x_col,
            y=y_col,
            hue="congruency",
            hue_order=hue_order,
            palette=palette,
            alpha=0.8,
            legend=False,
            ax=axes[j],
        )

        if var != "mean_acc":
            # Use empirical range for axis limits; handle all-NaN gracefully
            x_min = plot_data[x_col].min(skipna=True)
            x_max = plot_data[x_col].max(skipna=True)
            if pd.isna(x_min) or pd.isna(x_max):
                lims = [0.0, 1.0]
            else:
                lims = [float(x_min) - 0.02, float(x_max) + 0.02]
        else:
            lims = [0.6, 1.0]

        axes[j].plot(lims, lims, color="black", linestyle="--", linewidth=1)
        axes[j].set_xlim(lims)
        axes[j].set_ylim(lims)

        axes[j].set_xlabel("")
        axes[j].set_ylabel("")
        axes[j].set_title(titles[j], fontsize=fontsize - 5)

    fig.supxlabel("Empirical", fontsize=fontsize - 5, y=0.0)
    fig.supylabel("Resimulated", fontsize=fontsize - 5, x=0.0)
    fig.tight_layout()

    return fig, axes


def make_strictly_increasing(
    edges: Iterable[float],
    eps: float = 1e-9
) -> npt.NDArray[np.float64]:
    """
    Ensure a sequence of numeric values is strictly monotonically increasing.

    This function enforces strict monotonicity by scanning the input array
    from left to right and adjusting any non-increasing element so that it
    exceeds its predecessor by at least `eps`. The adjustment is performed
    in-place on a copied NumPy array, leaving the original input unchanged.

    This utility is particularly useful when constructing bin edges for
    histogramming or quantile-based discretization, where duplicate or
    non-increasing edges can cause downstream numerical or categorical
    binning errors.

    Parameters
    ----------
    edges : Iterable[float]
        A one-dimensional sequence of numeric values intended to represent
        ordered boundaries (e.g., histogram bin edges). The input does not
        need to be strictly increasing.
    eps : float, optional
        The minimum increment enforced between adjacent values when a
        violation of strict monotonicity is detected. Default is 1e-9.

    Returns
    -------
    numpy.ndarray
        A one-dimensional NumPy array of dtype float64 with strictly
        increasing values.

    Notes
    -----
    - The function guarantees `edges[i] > edges[i-1]` for all `i > 0`.
    - Adjustments are minimal and only applied when necessary.
    - The magnitude of `eps` should be chosen with respect to the numerical
      scale of `edges` to avoid unintended distortion.

    Examples
    --------
    >>> make_strictly_increasing([0.0, 1.0, 1.0, 2.0])
    array([0.0, 1.0, 1.000000001, 2.0])

    >>> make_strictly_increasing([3, 2, 1])
    array([3.0, 3.000000001, 3.000000002])
    """
    edges_array = np.asarray(edges, dtype=float).copy()

    for k in range(1, len(edges_array)):
        if edges_array[k] <= edges_array[k - 1]:
            edges_array[k] = edges_array[k - 1] + eps

    return edges_array


def get_bin_edges(
    rt: Iterable[float],
    quantiles: npt.ArrayLike = np.linspace(0.1, 0.9, 9),
) -> Optional[npt.NDArray[np.float64]]:
    """
    Construct strictly increasing quantile-based bin edges for response times.

    This function computes empirical quantiles of the provided response time
    (RT) sample and returns bin edges suitable for discretization (e.g.,
    histogramming or multinomial likelihood construction). The returned edges
    are bounded by negative and positive infinity to ensure full coverage of
    the support.

    Non-finite RT values (NaN, ±inf) are removed prior to quantile estimation.
    If no finite observations remain, the function returns None.

    To guard against numerical degeneracy (e.g., repeated quantiles due to ties),
    the resulting edges are passed through `make_strictly_increasing`, ensuring
    strict monotonicity.

    Parameters
    ----------
    rt : Iterable[float]
        One-dimensional sequence of response times. May contain non-finite
        values, which will be removed prior to quantile computation.
    quantiles : array-like, optional
        Sequence of quantile probabilities in the interval [0, 1] used to
        define internal bin boundaries. Default is nine equally spaced
        quantiles from 0.1 to 0.9 (inclusive), yielding ten bins.

    Returns
    -------
    numpy.ndarray or None
        A one-dimensional NumPy array of strictly increasing bin edges
        with the form:

            [-inf, q1, q2, ..., qK, +inf]

        where q1...qK are empirical quantiles of the finite RT values.

        Returns None if no finite RT observations are available.

    Notes
    -----
    - The number of resulting bins equals len(quantiles) + 1.
    - Quantile-based binning yields approximately equal expected counts
      per bin under the empirical distribution.
    - Strict monotonicity is enforced to prevent downstream errors in
      functions such as `pandas.cut` or histogram-based likelihood
      computations.
    - The function assumes `rt` represents a univariate distribution.

    Examples
    --------
    >>> rt = [0.35, 0.42, 0.51, 0.60, 0.72]
    >>> get_bin_edges(rt)
    array([-inf, 0.392, 0.434, ..., 0.688, inf])

    >>> get_bin_edges([np.nan, np.inf])
    None
    """
    rt_array = np.asarray(rt, dtype=float)
    rt_array = rt_array[np.isfinite(rt_array)]

    if rt_array.size == 0:
        return None

    q = np.quantile(rt_array, quantiles)
    edges = np.concatenate(([-np.inf], q, [np.inf]))

    return make_strictly_increasing(edges)

def count_bins(
    data: pd.DataFrame,
    bin_edges: Optional[npt.ArrayLike],
    part: Union[int, str],
    congruency: Hashable,
    congruency_condition: Union[int, str],
    accuracy: Hashable,
    accuracy_condition: Union[int, str],
    *,
    id_name: Hashable = "id",
    rt: Hashable = "rt",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin response times and count observations per bin for a single participant × condition cell.

    This helper constructs a complete RT-bin count table for a specific cell defined by
    `(part, congruency_condition, accuracy_condition)`. It returns a DataFrame with exactly
    `n_bins` rows (rt_bin = 0..n_bins-1) even when some bins contain zero observations.

    The function is designed for quantile-binning workflows (e.g., multinomial likelihood /
    G² deviance computations) where both observed and simulated data must be represented on
    an identical set of RT bin edges.

    Workflow:
    1) Create a full "skeleton" DataFrame (`empty_df`) containing all bin indices for the
       requested cell.
    2) Filter non-finite RT values (NaN, ±inf).
    3) Use `pandas.cut` to assign each RT to an integer bin index based on `bin_edges`.
    4) Count observations in each bin via `groupby(...).count()`.
    5) Left-merge counts into the skeleton and fill missing bins with zeros.

    Parameters
    ----------
    data : pandas.DataFrame
        Trial-level data for (at least) one participant and one congruency × accuracy cell.
        Must contain columns referenced by `id_name`, `congruency`, `accuracy`, and `rt`.
    bin_edges : array-like or None
        Bin edges to use for RT discretization. Typically produced by `get_bin_edges(...)`.
        Must be strictly increasing and compatible with `pandas.cut`.
        If None, or if `data` has zero rows, the function returns zero counts for all bins.
    part : int or str
        Participant identifier to populate in the returned count table. This value is written
        into the `id_name` column for all returned rows.
    congruency : Hashable
        Column name in `data` which stores congruency labels (e.g., "congruent"/"incongruent").
        Also used as the column name in the returned DataFrame.
    congruency_condition : int or str
        Condition value to populate in the returned congruency column for all rows (e.g.,
        "congruent").
    accuracy : Hashable
        Column name in `data` which stores accuracy coding (e.g., 0/1).
        Also used as the column name in the returned DataFrame.
    accuracy_condition : int or str
        Accuracy value to populate in the returned accuracy column for all rows (e.g., 1 for
        correct, 0 for error).
    id_name : Hashable, optional
        Column name for participant IDs in `data` and in the returned DataFrame.
        Default is "id".
    rt : Hashable, optional
        Column name in `data` containing response times to be binned. Default is "rt".
    n_bins : int, optional
        Number of RT bins expected (i.e., the number of intervals implied by `bin_edges`).
        The returned DataFrame will contain exactly `n_bins` rows with `rt_bin = 0..n_bins-1`.
        Default is 10.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns `[id_name, congruency, accuracy, "rt_bin", "obs_count"]`
        and exactly `n_bins` rows. `obs_count` is an integer count (may be returned as float
        after merge/fill operations; cast if you need strict integer dtype).

    Notes
    -----
    - RT values outside the interior edges are still assigned a bin because `bin_edges`
      is typically bounded by [-inf, +inf].
    - Trials whose RT cannot be binned (e.g., due to malformed edges) are dropped via
      `dropna(subset=["rt_bin"])`.
    - This function does not validate that `bin_edges` implies `n_bins`; ensure consistency
      upstream (e.g., `n_bins = len(bin_edges) - 1`).

    Examples
    --------
    >>> edges = get_bin_edges(obs_cell["rt"])  # [-inf, q1, ..., q9, inf]
    >>> counts = count_bins(
    ...     data=obs_cell,
    ...     bin_edges=edges,
    ...     part=12,
    ...     congruency="congruency",
    ...     congruency_condition="incongruent",
    ...     accuracy="accuracy",
    ...     accuracy_condition=1,
    ...     n_bins=10,
    ... )
    >>> counts.head()
       id    congruency  accuracy  rt_bin  obs_count
    0  12  incongruent         1       0          3
    1  12  incongruent         1       1          4
    """
    empty_df = pd.DataFrame(
        {
            id_name: part,
            congruency: congruency_condition,
            accuracy: accuracy_condition,
            "rt_bin": range(n_bins),
        }
    )

    if data.shape[0] == 0 or bin_edges is None:
        empty_df["obs_count"] = 0
        return empty_df

    data = data.copy()
    data = data[np.isfinite(data[rt])]

    data["rt_bin"] = pd.cut(data[rt], bins=bin_edges, labels=False, include_lowest=True)
    data = data.dropna(subset=["rt_bin"])
    data["rt_bin"] = data["rt_bin"].astype(int)

    count_data = (
        data.groupby([id_name, congruency, accuracy, "rt_bin"])[rt]
        .count()
        .reset_index(name="obs_count")
    )

    return (
        empty_df.merge(
            count_data, on=[id_name, congruency, accuracy, "rt_bin"], how="left"
        )
        .fillna(0)
    )

def goodness_of_fit(
    observed: npt.ArrayLike,
    expected: npt.ArrayLike,
    statistic: Literal["g2", "chi2", "both"] = "both",
) -> float | tuple[float, float]:
    """
    Compute multinomial goodness-of-fit statistics (G² and/or Pearson χ²).

    This function evaluates the discrepancy between observed counts (O_j)
    and model-implied expected counts (E_j) using:

        - Likelihood-ratio deviance (G²):
            G² = 2 Σ O_j log(O_j / E_j)

        - Pearson chi-square (χ²):
            χ² = Σ (O_j − E_j)² / E_j

    Both statistics are asymptotically chi-square distributed under
    regularity conditions, with degrees of freedom equal to the number
    of independent cells minus the number of fitted parameters.

    Parameters
    ----------
    observed : array-like
        One-dimensional array of non-negative observed counts (O_j).
    expected : array-like
        One-dimensional array of strictly positive expected counts (E_j).
        Must have the same shape as `observed`.
    statistic : {"g2", "chi2", "both"}, optional
        Which statistic to return:
            - "g2"   → return G² only 
            - "chi2" → return Pearson χ² only
            - "both" → return (G², χ²) (default)

    Returns
    -------
    float or tuple of float
        The requested goodness-of-fit statistic(s).
        Returns np.inf if any E_j <= 0 where O_j > 0 (G² undefined)
        or if any E_j <= 0 (χ² undefined).

    Raises
    ------
    ValueError
        If `observed` and `expected` differ in shape.

    Notes
    -----
    - Cells with O_j = 0 contribute 0 to G².
    - Pearson χ² includes all cells (including O_j = 0).
    - No continuity correction is applied.
    - Assumes multinomial count structure.

    Examples
    --------
    >>> goodness_of_fit([10, 15, 5], [12, 12, 6], "g2")
    1.527...

    >>> goodness_of_fit([10, 15, 5], [12, 12, 6], "chi2")
    1.583...

    >>> goodness_of_fit([10, 15, 5], [12, 12, 6], "both")
    (1.527..., 1.583...)
    """
    O = np.asarray(observed, dtype=float)
    E = np.asarray(expected, dtype=float)

    if O.shape != E.shape:
        raise ValueError("`observed` and `expected` must have the same shape.")

    if np.any(E <= 0):
        return float(np.inf) if statistic != "both" else (float(np.inf), float(np.inf))

    # --- G² ---
    mask = O > 0
    G2 = 2.0 * np.sum(O[mask] * np.log(O[mask] / E[mask]))

    # --- Pearson χ² ---
    chi2 = np.sum((O - E) ** 2 / E)

    if statistic == "g2":
        return float(G2)
    elif statistic == "chi2":
        return float(chi2)
    elif statistic == "both":
        return float(G2), float(chi2)
    else:
        raise ValueError("`statistic` must be one of {'g2', 'chi2', 'both'}.")

def compute_gof(
    data_obs: pd.DataFrame,
    data_model: pd.DataFrame,
    *,
    rt: Hashable = "rt",
    congruency: Hashable = "congruency",
    accuracy: Hashable = "accuracy",
    min_n_err: int = 5,
    id_name: Hashable = "id",
    n_bins: int = 10,
    E_min: int = 1
) -> npt.NDArray[np.float64]:
    """
    Compute cell-wise multinomial G² deviances between observed and model data.

    This function evaluates model fit by comparing observed and simulated
    response-time (RT) distributions within each participant × congruency ×
    accuracy cell. RTs are discretized into quantile-based bins derived from
    the observed data, and a multinomial likelihood-ratio deviance (G²) is
    computed for each cell.

    For each participant and each condition combination:

        1. Empirical RT quantiles define bin edges.
        2. Observed and simulated RTs are binned using identical edges.
        3. Observed bin counts (O) are compared to model-implied expected
           counts (E) derived from simulated counts using Dirichlet smoothing.
        4. A G² deviance statistic is computed.

    The function returns a vector of G² values across all evaluable cells.

    Parameters
    ----------
    data_obs : pandas.DataFrame
        Observed trial-level dataset. Must contain at least:
        - participant identifier column (`id_name`)
        - congruency column (`congruency`)
        - accuracy column (`accuracy`)
        - RT column (`rt`)
    data_model : pandas.DataFrame
        Model-generated (simulated) trial-level dataset with the same column
        structure as `data_obs`.
    rt : Hashable, optional
        Column name for response times. Default is "rt".
    congruency : Hashable, optional
        Column name for congruency condition labels. Default is "congruency".
    accuracy : Hashable, optional
        Column name for accuracy coding (e.g., 0 = error, 1 = correct).
        Default is "accuracy".
    min_n_err : int, optional
        Minimum number of observed error trials required to compute a
        meaningful error RT distribution. Cells with fewer error trials
        are skipped. Default is 5.
    id_name : Hashable, optional
        Column name identifying participants. Default is "id".
    n_bins : int, optional
        Number of RT bins (typically quantile-based). Must be consistent
        with the output of `get_bin_edges`. Default is 10.
    E_min : int
        Minimum expected values in a bin. Is used to filter overestimated Chi² values. Default is 1.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of G² deviance values (float64), one per
        evaluable participant × congruency × accuracy cell.

    Notes
    -----
    - Quantile bin edges are computed from observed RTs only.
    - Model bin probabilities are estimated from simulated counts with
      Dirichlet smoothing (α = 0.5).
    - If the model produces zero simulated trials in a cell, the cell
      is skipped rather than forcing infinite deviance.
    - Cells with insufficient observed error trials (if accuracy == 0)
      are skipped.
    - The resulting G² values correspond to multinomial deviances on
      binned RT distributions, not to the continuous-time diffusion
      model likelihood.

    Statistical Interpretation
    --------------------------
    For each cell, the deviance is:

        G² = 2 * Σ O_j log(O_j / E_j),

    where O_j are observed bin counts and E_j are expected counts implied
    by the model. Under regularity conditions and large samples, G² is
    asymptotically chi-square distributed.

    Examples
    --------
    >>> g2_values = compute_g2(
    ...     data_obs=empirical_df,
    ...     data_model=simulated_df,
    ...     n_bins=10
    ... )
    >>> g2_values.mean()
    12.47
    """
    check_vars(data=data_obs, id_name=id_name, rt=rt,
               congruency=congruency, accuracy=accuracy)
    check_vars(data=data_model, id_name=id_name, rt=rt,
               congruency=congruency, accuracy=accuracy)

    data_obs = check_congruency(
        data=data_obs,
        rt=rt,
        congruency=congruency,
        output_coding_con="congruent",
        output_coding_inc="incongruent",
    )

    data_model = check_congruency(
        data=data_model,
        rt=rt,
        congruency=congruency,
        output_coding_con="congruent",
        output_coding_inc="incongruent",
    )

    parts = data_obs[id_name].unique()
    rows = []

    for idx in tqdm(range(0, len(parts)), desc=f"Compute Goodness of Fit"):
        for con in ["congruent", "incongruent"]:
            for acc in [0, 1]:

                part = parts[idx]

                obs_cell = data_obs[
                    (data_obs[id_name] == part)
                    & (data_obs[congruency] == con)
                    & (data_obs[accuracy] == acc)
                ].copy()

                mod_cell = data_model[
                    (data_model[id_name] == part)
                    & (data_model[congruency] == con)
                    & (data_model[accuracy] == acc)
                ].copy()

                if obs_cell.shape[0] == 0:
                    continue

                bin_edges = get_bin_edges(obs_cell[rt])

                count_obs = count_bins(
                    obs_cell,
                    bin_edges,
                    part,
                    id_name=id_name,
                    congruency=congruency,
                    accuracy=accuracy,
                    congruency_condition=con,
                    accuracy_condition=acc,
                    n_bins=n_bins,
                )

                count_mod = count_bins(
                    mod_cell,
                    bin_edges,
                    part,
                    id_name=id_name,
                    congruency=congruency,
                    accuracy=accuracy,
                    congruency_condition=con,
                    accuracy_condition=acc,
                    n_bins=n_bins,
                )

                merged = (
                    pd.merge(
                        count_obs,
                        count_mod,
                        on=[id_name, congruency, accuracy, "rt_bin"],
                        suffixes=["_obs", "_model"],
                        how="outer",
                    )
                    .fillna(0)
                )

                O = merged["obs_count_obs"].to_numpy(dtype=float)
                C = merged["obs_count_model"].to_numpy(dtype=float)

                N_obs = O.sum()
                N_sim = C.sum()

                if acc == 0 and N_obs < min_n_err:
                    continue

                if N_sim == 0:
                    continue

                alpha = 0.5
                J = len(C)

                pi = (C + alpha) / (N_sim + J * alpha)
                E = N_obs * pi

                G2, chi2 = goodness_of_fit(O, E, statistic="both")

                if np.min(E) < E_min:
                    chi2 = np.nan

                rows.append({
                    str(id_name): part,
                    str(congruency): con,
                    str(accuracy): acc,
                    "n_obs": float(N_obs),
                    "n_sim": float(N_sim),
                    "g2": float(G2),
                    "chi2": float(chi2),
                    "min_E": float(np.min(E)),
                })

    return pd.DataFrame(rows)