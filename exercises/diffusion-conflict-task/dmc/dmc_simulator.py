"""
Code written by Simon Schaefer.
Source: https://github.com/simschaefer/amortized-dmc
"""

import numpy as np
from scipy.stats import truncnorm
import warnings

class DMC:
    def __init__(
        self,
        prior_means: np.ndarray,
        prior_sds: np.ndarray,
        param_names: tuple[str] = ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r'),
        param_lower_bound: float | None = 0,
        fixed_num_obs: int | None = 200,
        tmax: int = 1200,
        dt: float = 1.0,
        sigma: float = 4.0,
        X0_beta_shape_fixed: float = 3,
        sdr_fixed: float | None = None,
        a_value: int = 2,
        num_conditions: int = 2,
        contamination_probability: float | None = None,
        contamination_uniform_lower: float = 0,
        contamination_uniform_upper: float = 2,
        min_num_obs: int = 50,
        max_num_obs: int = 800,
        rng: np.random.Generator | None = None
    ):
        """
        Initialize the DMC simulator in a BayesFlow-friendly format.

        Parameters
        ----------
        prior_means : np.ndarray
            Array of prior means for the model parameters. 
        prior_sds : np.ndarray
            Array of prior standard deviations for the model parameters.
        param_names : tuple of str, optional
            Names of the parameters. Default is ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r').
        param_lower_bound : float or None, optional
            Lower bound for the prior.
        fixed_num_obs : float, optional
            Number of simulated trials. Default is 200. Set to None and specify minimal number of observations (min_num_obs) and
            maximum number of observations (max_num_obs) to include random sampling of trial numbers.
        min_num_obs:
            Lower boundary of uniform distribution used to randomly sample trials numbers. Only applied if fixed_num_obs is not None.
        max_num_obs:
            Upper boundary of uniform distribution used to randomly sample trials numbers. Only applied if fixed_num_obs is not None.
        tmax : int, optional
            Maximum simulation time in milliseconds. Default is 1200.
        dt : float, optional
            Time step of the simulation. Default is 1.
        sigma : floaFixed value for trial-to-trial variability in non-decision time.
            If None, the non-decision time is drawn from a normal distribution with standard deviation sdr drawn from specified prior distributions.t, optional
            Standard deviation of the noise in the diffusion process. Default is 4.0.
        X0_beta_shape_fixed : float, optional
            Shape parameter used for beta distribution of initial states. Default is 3.0.
        sdr_fixed: float, None
            Fixed value for trial-to-trial variability in non-decision time.
            If None, the non-decision time is sampled from a normal distribution with standard deviation sdr, which is itself drawn from the specified prior distribution.
        a_value : int, optional
            Constant 'a' value used in the simulation. Default is 2. a > 1 is a necessary condition to simulate data.
        num_conditions : int, optional
            The number of conditions in the experiment. Default is 2.
        contamination_probability :
            Rate of contamination during robust training. 
        contamination_uniform_lower:
            lower bound of random RTs used in robust training.
        contamination_uniform_upper:
            upper bound of random RTs used in robust training.
        """

        self.fixed_num_obs = fixed_num_obs
        self.tmax = tmax
        self.dt = dt
        self.sigma = sigma
        self.param_lower_bound = param_lower_bound
        self.X0_beta_shape_fixed = X0_beta_shape_fixed
        self.a_value = a_value
        self.num_conditions = num_conditions
        self.contamination_probability = contamination_probability
        self.contamination_uniform_lower = contamination_uniform_lower
        self.contamination_uniform_upper = contamination_uniform_upper
        self.min_num_obs = min_num_obs
        self.max_num_obs = max_num_obs
        self.sdr_fixed = sdr_fixed
        self.rng = rng if rng is not None else np.random.default_rng()

        if prior_means.ndim != 1 or prior_sds.ndim != 1:
            raise ValueError("prior_means and prior_sds must be 1D arrays.")

        n_means = prior_means.shape[0]
        n_sds = prior_sds.shape[0]

        if n_means != n_sds:
            raise ValueError(
                f"prior_means and prior_sds must have the same length, got {n_means} and {n_sds}."
            )
    
        if n_means != len(param_names):
            raise ValueError(
                f"Expected {len(param_names)} prior entries to match param_names, got {n_means}."
            )

        allowed_names = {'A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r'}

        # basic structure checks
        if not isinstance(param_names, tuple):
            raise TypeError("param_names must be a tuple of parameter names.")

        if len(param_names) != len(set(param_names)):
            raise ValueError(f"param_names contains duplicates: {param_names}")

        if not set(param_names).issubset(allowed_names):
            invalid = tuple(sorted(set(param_names) - allowed_names))
            raise ValueError(
                f"param_names contains invalid entries {invalid}. "
                f"Allowed names are {tuple(sorted(allowed_names))}."
            )

        # determine which names are required given sdr_fixed
        if self.sdr_fixed is None:
            required_names = {'A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r'}
        else:
            required_names = {'A', 'tau', 'mu_c', 'mu_r', 'b'}

        if set(param_names) != required_names:
            raise ValueError(
                f"For sdr_fixed={self.sdr_fixed}, param_names must contain exactly "
                f"{tuple(sorted(required_names))}. Got {param_names}."
            )

        self.prior_means = prior_means
        self.prior_sds = prior_sds

        if self.a_value <= 1:
            raise ValueError(f"a (gamma shape) = {a_value}. Please choose a value larger than 1.")

        if num_conditions != 2:
            raise ValueError("Number of conditions must be 2 for this experiment.")
        
        if np.any(prior_sds <= 0):
            raise ValueError("All prior_sds must be strictly positive.")

        if dt <= 0:
            raise ValueError("dt must be > 0.")

        if tmax <= 0:
            raise ValueError("tmax must be > 0.")

        if self.fixed_num_obs is not None:
            if not isinstance(self.fixed_num_obs, (int, np.integer)) or self.fixed_num_obs <= 0:
                raise ValueError("fixed_num_obs must be a positive integer or None.")

        if min_num_obs <= 0 or max_num_obs <= 0 or min_num_obs > max_num_obs:
            raise ValueError("Require 0 < min_num_obs <= max_num_obs.")

        if contamination_probability is not None and not (0 <= contamination_probability <= 1):
            raise ValueError("contamination_probability must be between 0 and 1.")

        if self.contamination_uniform_lower > self.contamination_uniform_upper:
            raise ValueError(
                "contamination_uniform_lower must be <= contamination_uniform_upper."
            )
    
        if sdr_fixed is not None and sdr_fixed < 0:
            raise ValueError("sdr_fixed must be >= 0.")
        
        self.param_names = tuple(param_names)
        self.prior_means = prior_means
        self.prior_sds = prior_sds

    def prior(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """
        Sample model parameters from the prior distribution.

        Parameters
        ----------
        rng : np.random.Generator or None, optional
            Random number generator used for sampling. If None, the instance's
            default generator (`self.rng`) is used.

        Returns
        -------
        dict[str, float]
            A dictionary mapping parameter names (`self.param_names`) to sampled
            values.

        Notes
        -----
        The prior distribution is defined independently for each parameter as a
        normal distribution with mean `self.prior_means` and standard deviation
        `self.prior_sds`.

        If `self.param_lower_bound` is not None, the normal distribution is
        truncated from below at this value. In that case, samples are drawn from a
        lower-truncated normal distribution using `scipy.stats.truncnorm`.

        Otherwise, samples are drawn from an unconstrained normal distribution.

        The truncation is applied element-wise, i.e., each parameter shares the
        same lower bound but retains its own mean and standard deviation.

        Examples
        --------
        >>> params = simulator.prior()
        >>> params
        {'A': 1.23, 'tau': 45.6, 'mu_c': 0.12, 'mu_r': 210.3, 'b': 90.1, 'sd_r': 15.2}
        """

        rng = rng if rng is not None else self.rng

        if self.param_lower_bound is not None:
            a = (self.param_lower_bound - self.prior_means) / self.prior_sds
            b = (np.inf - self.prior_means) / self.prior_sds
            p = truncnorm.rvs(a, b, loc=self.prior_means, scale=self.prior_sds, random_state=rng)
        else:
            p = rng.normal(self.prior_means, self.prior_sds)
    
        return dict(zip(self.param_names, p))


    def trial(self, 
              A: float, 
              tau: float, 
              mu_c: float, 
              b: float, 
              t: np.ndarray, 
              noise: np.ndarray, 
              non_decision_ts: np.ndarray,
              rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate multiple DMC trials in parallel.

        This function generates decision trajectories for multiple trials using
        an Euler–Maruyama discretization of a time-varying drift-diffusion process.
        For each trial, the process evolves until it crosses an upper (+b) or lower
        (-b) decision boundary, or until the time grid is exhausted.

        Parameters
        ----------
        A : float
            Amplitude of the time-dependent control signal.
        tau : float
            Time constant governing the shape of the control signal.
        mu_c : float
            Constant (baseline) drift component.
        b : float
            Symmetric decision boundary. Upper boundary is +b, lower boundary is -b.
        t : np.ndarray, shape (T,)
            Array of discrete time points defining the simulation grid.
        noise : np.ndarray, shape (n_trials, T)
            Gaussian noise samples for all trials and time steps. Each row corresponds
            to one trial.
        non_decision_ts : np.ndarray, shape (n_trials,)
            Non-decision times (in milliseconds) for each trial.
        rng : np.random.Generator or None, optional
            Random number generator used for sampling initial states. If None,
            `self.rng` is used.

        Returns
        -------
        np.ndarray, shape (n_trials, 2)
            A 2D array where:
            - column 0 contains response times (RTs) in seconds
            - column 1 contains responses:
                1  → upper boundary reached (correct)
                0  → lower boundary reached (error)
            -1 → no boundary crossing within the time window

            Trials without a boundary crossing receive RT = -1 and response = -1.
        """

        rng = rng if rng is not None else self.rng

        num_trials, _ = noise.shape
        dt = self.dt
        sqrt_dt_sigma = self.sigma * np.sqrt(dt)

        # Initial positions X0 for all trials
        X0 = rng.beta(self.X0_beta_shape_fixed, self.X0_beta_shape_fixed, size=num_trials) * (2 * b) - b

        # Drift term mu(t), shape (T,)

        if self.a_value != 2:
            t_div_tau = t / tau
            exponent_term = np.exp(-t_div_tau)
            power_term = (np.exp(1) * t_div_tau / (self.a_value - 1)) ** (self.a_value - 1)
            deriv_term = ((self.a_value - 1) / t) - (1 / tau)
            mu_t = A * exponent_term * power_term * deriv_term + mu_c  # shape (T,)
        
        else:
            mu_t = A/tau * np.exp(1 - t/tau) * (1 - t/tau) + mu_c

        # Full drift for all trials: broadcast mu_t to (n_trials, T)
        dX = mu_t[None, :] * dt + sqrt_dt_sigma * noise  # shape (n_trials, T)
        X_shift = np.cumsum(dX, axis=1) + X0[:, None]    # shape (n_trials, T)

        # Check boundary crossings
        crossed_upper = X_shift >= b
        crossed_lower = X_shift <= -b
        crossed_any = crossed_upper | crossed_lower

        # First crossing index for each trial
        first_crossing = np.argmax(crossed_any, axis=1)
        has_crossed = np.any(crossed_any, axis=1)

        # Prepare output
        rts = np.full(num_trials, -1.0)
        resps = np.full(num_trials, -1)

        # Fill only for trials that crossed
        idx = np.where(has_crossed)[0]
        crossing_times = t[first_crossing[idx]]

        # use nondecision times only for trials that crossed
        non_decision_ts_crossed = non_decision_ts[idx]

        rts[idx] = (crossing_times + non_decision_ts_crossed) / 1000  # convert to seconds

        # Determine response type
        resp_hit = X_shift[idx, first_crossing[idx]]
        resps[idx] = (resp_hit >= b).astype(int)

        return np.c_[rts, resps]
    
    def experiment(
        self, 
        A: float, 
        tau: float, 
        mu_c: float, 
        mu_r: float, 
        b: float,
        num_obs: int,
        sd_r: float = 0,
        rng: np.random.Generator | None = None
    )-> dict[str, np.ndarray | int]:
        """
        Simulate a full DMC experiment consisting of multiple trials across conditions.

        This function generates reaction times and responses for a specified number
        of trials, split evenly across experimental conditions (e.g., congruent vs.
        incongruent). Each trial is simulated using the `trial()` method.

        Parameters
        ----------
        A : float
            Amplitude of the control signal. The sign is flipped across conditions.
        tau : float
            Time constant governing the shape of the control signal.
        mu_c : float
            Constant drift component.
        mu_r : float
            Mean non-decision time (in milliseconds).
        b : float
            Symmetric decision boundary.
        num_obs : int
            Total number of trials to simulate.
        sd_r : float, optional
            Standard deviation of non-decision time across trials. If `self.sdr_fixed`
            is not None, that value overrides this argument.
        rng : np.random.Generator or None, optional
            Random number generator used for sampling noise and non-decision times.
            If None, `self.rng` is used.

        Returns
        -------
        dict[str, np.ndarray | int]
            A dictionary with the following entries:

            - "rt" : np.ndarray, shape (num_obs,)
                Reaction times in seconds. Trials without a boundary crossing are
                assigned -1.
            - "accuracy" : np.ndarray, shape (num_obs,)
                Binary responses (1 = upper boundary, 0 = lower boundary, -1 = no response).
            - "conditions" : np.ndarray, shape (num_obs,)
                Condition labels (0 = congruent, 1 = incongruent).
            - "num_obs" : int
                Total number of simulated trials.
        """

        rng = rng if rng is not None else self.rng
        
        # random number of trials
        # num_obs = self.num_obs or np.random.randint(min_num_obs, max_num_obs+1)
        # sd_r fixed or sampeld from prior-function
        sd_r = self.sdr_fixed if self.sdr_fixed is not None else sd_r
        
        # congruency conditions (equal split)
        obs_per_condition = int(np.ceil(num_obs / self.num_conditions))
        conditions = np.repeat(np.arange(self.num_conditions), obs_per_condition)

        # precompute vector of time steps and 2D-noise
        t = np.arange(self.dt, self.tmax + self.dt, self.dt)
        T = len(t)

        noise = rng.normal(size=(num_obs, T))
        non_decision_ts = rng.normal(size=num_obs, loc=mu_r, scale=sd_r)
        
        data = np.zeros((num_obs, 2))
        
        # simulate CONGRUENT trials (positive Amplitude A)
        data[:obs_per_condition] = self.trial(
            A=A, tau=tau, mu_c=mu_c, b=b, t=t, noise=noise[:obs_per_condition], non_decision_ts=non_decision_ts[:obs_per_condition], rng=rng
        )
        
        # simulate INCONGRUENT trials (negative Amplitude A)
        data[obs_per_condition:] = self.trial(
            A=-A, tau=tau, mu_c=mu_c, b=b, t=t, noise=noise[obs_per_condition:], non_decision_ts=non_decision_ts[obs_per_condition:], rng=rng,
        )
        
        conditions = conditions[:num_obs]
        
        # include contamination if probability is given
        if self.contamination_probability is not None and self.contamination_probability > 0:
            
            # compute binomial mask with given contamination probability
            binom_mask = rng.binomial(1, p=self.contamination_probability, size=num_obs) == 1

            # replace RTs by uniform samples
            data[:,0][binom_mask] = rng.uniform(self.contamination_uniform_lower, 
                                                      self.contamination_uniform_upper, 
                                                      size=np.sum(binom_mask))

            # replace responses by random choices
            data[:,1][binom_mask] = rng.binomial(1, p=0.5, size=np.sum(binom_mask))

        return dict(rt=data[:, 0], accuracy=data[:, 1], conditions=conditions, num_obs=num_obs)

    def sample(self, 
               batch_size: int, 
               num_obs: int | None = None, 
               seed: int | None = None,
               **kwargs
               ) -> dict[str, np.ndarray]:
        """
        Generate a batch of simulated datasets together with their corresponding
        parameter values.

        This function repeatedly samples parameters from the prior distribution and
        simulates datasets via the model, returning stacked arrays.

        Parameters
        ----------
        batch_size : int or tuple
            Number of independent datasets to simulate. If a tuple is provided,
            only the first element is used.
        num_obs : int or None, optional
            Number of observations (trials) per dataset. If None:
            - `self.fixed_num_obs` is used if specified, or
            - a random number of trials is drawn uniformly from
            [`self.min_num_obs`, `self.max_num_obs`].
        seed : int or None, optional
            Seed for the random number generator. If provided, a new generator is
            initialized to ensure reproducibility. Otherwise, `self.rng` is used.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing simulated parameters and observables, where each
            entry has shape `(batch_size, ..., 1)` with keys:

            - parameter names (e.g., "A", "tau", ...)
            - "rt" : reaction times
            - "accuracy" : binary responses
            - "conditions" : condition labels
            - "num_obs" : number of trials per dataset

            The trailing singleton dimension is added to facilitate concatenation
            with other datasets.

        Notes
        -----
        Each dataset is generated by calling the model instance (`self(...)`), which:
        1. Samples parameters from the prior.
        2. Simulates trial-level data using `experiment()`.

        All simulations in the batch share the same random number generator, ensuring
        reproducibility when a seed is provided.

        Examples
        --------
        >>> sims = simulator.sample(batch_size=32, num_obs=200, seed=42)
        >>> sims["rt"].shape
        (32, 200, 1)
        >>> sims["A"].shape
        (32, 1)
        """

        rng = np.random.default_rng(seed) if seed is not None else self.rng

        if num_obs is None:
            if self.fixed_num_obs is not None:
                num_obs = self.fixed_num_obs
            else:
                num_obs = int(rng.integers(self.min_num_obs, self.max_num_obs + 1))
        
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        sims = [self(num_obs, rng=rng) for _ in range(batch_size)]
        sims = {k: np.stack([s[k] for s in sims], axis=0) for k in sims[0].keys()}

        # Ensure everything has a trailing dimension of 1 (so its concateneable)
        sims = {k: v[..., np.newaxis] for k, v in sims.items()}

        return sims

    def __call__(self, 
                 num_obs: int | None,
                 rng: np.random.Generator | None = None,
                 **kwargs
                 )-> dict[str, np.ndarray | float | int]:
        
        """
        Generate a single simulated dataset together with its underlying parameter values.

        This method provides a convenient interface to:
        1. Sample a parameter vector from the prior distribution.
        2. Simulate trial-level data using these parameters.

        It is the core function used by `sample()` to generate batches of simulations.

        Parameters
        ----------
        num_obs : int or None
            Number of observations (trials) to simulate. If None, the number of trials
            must be handled internally by `experiment()` (e.g., via fixed or randomly
            sampled trial counts).
        rng : np.random.Generator or None, optional
            Random number generator used for both parameter sampling and data simulation.
            If None, `self.rng` is used.
        **kwargs
            Additional keyword arguments (currently unused, included for API compatibility).

        Returns
        -------
        dict[str, np.ndarray | float | int]
            A dictionary containing both sampled parameters and simulated data.
            The output merges the prior draws and simulation results, typically including:

            - parameter names (e.g., "A", "tau", "mu_c", "mu_r", "b", "sd_r")
            - "rt" : np.ndarray, shape (num_obs,)
                Reaction times in seconds.
            - "accuracy" : np.ndarray, shape (num_obs,)
                Binary responses (1 = upper boundary, 0 = lower boundary, -1 = no response).
            - "conditions" : np.ndarray, shape (num_obs,)
                Condition labels.
            - "num_obs" : int
                Number of simulated trials.

        Notes
        -----
        - The returned dictionary is formed by merging (`|`) the parameter dictionary
        from `prior()` and the simulation output from `experiment()`.
        - Parameter values are scalar floats, while simulated observables are arrays.
        - Using a shared `rng` ensures reproducibility across both parameter sampling
        and data generation.

        Examples
        --------
        >>> sim = simulator(num_obs=200, rng=np.random.default_rng(42))
        >>> sim["rt"].shape
        (200,)
        >>> sim["A"]
        0.87
        """
        
        rng = rng if rng is not None else self.rng

        # draw priors
        prior_draws = self.prior(rng=rng)

        # run experiment
        sims = self.experiment(**(prior_draws), num_obs=num_obs, rng=rng)

        return prior_draws | sims


