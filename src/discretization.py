"""
Author - Philip Ciunkiewicz

This module provides discretization classes for
both dynamic (as proposed by Fenton and Neil) and
static strategies.
"""
from abc import ABC, abstractmethod
from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer


class Discretizer(ABC):
    """
    Abstract base discretization class.
    """
    def __init__(self):
        self.edges = None
        self.intervals = None

    def _compute_intervals(self):
        """
        Convenience method for computing the interval bounds
        from the provided interval edges. Intervals are re-
        computed whenever edges are introduced through splitting
        or removed though merging.
        """
        self.intervals = [(x, self.edges[i+1])
                          for i, x in enumerate(self.edges[:-1])]

    def discretize(self, x):
        """
        Discretizes continuous values using bisection.

        Params
        ------
        x: float|int
            Continuous value to be discretized.

        Returns
        -------
        interval: tuple
            Corresponding discrete interval for `x`.
        """
        idx = bisect_left(self.edges, x)
        return self.intervals[max(0, idx-1)]

    def transform(self, data=None):
        """
        Transform data by applying discretization and
        string conversion to values.

        Params
        ------
        data: pandas.Series (optional)
            Optional data to discretize, default uses self.X.

        Returns
        -------
        data: pandas.Series
            Discretized intervals corresponding to data.
        """
        if data is not None:
            return data.apply(self.discretize).apply(self._stringify)
        return self.X.apply(self.discretize).apply(self._stringify)

    @abstractmethod
    def _stringify(self, interval, ndigits=3):
        """
        Convenience method for representing an interval using
        mathematical notation with a closed lower bound and
        open upper bound.

        Params
        ------
        interval: tuple
            Interval edges of shape (2,).

        ndigits: int (optional)
            Number of significant digits for rounding values.

        Returns
        -------
        interval: string
            Interval bounds in mathematical notation.
        """
        pass


class DynamicDiscretizer(Discretizer):
    """
    Dynamic Discretization class for Bayesian networks.

    Discretizes continuous variables based on information
    theoretic measures, specifically relative entropy
    error (special case of KL divergence).
    """
    def __init__(self, X, bins=4, edges=None, kde_kws={}):
        """
        Params
        ------
        X: pandas.Series
            The feature you wish to discretize.

        bins: int (optional)
            The number of initial uniform discrete intervals.
            If `edges` is provided this value is ignored.

        edges: array_like (optional)
            Array or list of initial discrete interval edges.

        kde_kws: dict (optional)
            Keyword arguments to pass to scipy.stats.gaussian_kde.
        """
        super().__init__()

        if edges:
            self.edges = list(edges)
        else:
            self.edges = np.linspace(X.min(), X.max(), bins+1).tolist()

        self.X = X.copy()
        self.kde = stats.gaussian_kde(X, **kde_kws)
        self.errors = []

        self.initial = self.edges.copy()
        self._compute_intervals()

    def _compute_error(self, interval):
        """
        Compute relative entropy error on a given interval.

        Params
        ------
        interval: tuple
            Interval edges of shape (2,).

        Returns
        -------
        error: float
            Relative entropy error on the interval.
        """
        fmin = self.kde(interval[0])
        fmid = self.kde(np.mean(interval))
        fmax = self.kde(interval[1])

        term1 = (fmax - fmid) / (fmax - fmin) * fmin * np.log(fmin / fmid)
        term2 = (fmid - fmin) / (fmax - fmin) * fmax * np.log(fmax / fmid)

        return np.abs(term1 + term2)[0] * np.ptp(interval)

    def _split(self):
        """
        Logic for splitting the interval containing the highest
        relative entropy error. Split occurs at the midpoint of
        the interval and new intervals are computed.
        """
        err = [self._compute_error(x) for x in self.intervals]
        idx = np.argmax(err)
        self.edges.insert(idx+1, np.mean(self.intervals[idx]))
        self._compute_intervals()

    def _merge(self, beta):
        """
        Logic for merging neighboring intervals with sufficiently
        low relative entropy error. Intervals are merged only if
        >5 total intervals are present and two or more consecutive
        intervals have low error.

        Params
        ------
        beta: float
            Error threshold for merging consecutive intervals.
        """
        if len(self.intervals) > 5:
            err = [self._compute_error(x) for x in self.intervals]
            idx = np.argwhere(np.array(err) < beta).flatten()[::-1]
            for i, diff in enumerate(np.abs(np.diff(idx))):
                if diff == 1:
                    self.edges.pop(idx[i])
            self._compute_intervals()

    def _stringify(self, interval, ndigits=3):
        """
        Convenience method for representing an interval using
        mathematical notation with a closed lower bound and
        open upper bound.

        Params
        ------
        interval: tuple
            Interval edges of shape (2,).

        ndigits: int (optional)
            Number of significant digits for rounding values.

        Returns
        -------
        interval: string
            Interval bounds in mathematical notation.
        """
        lower, upper = [round(bound, ndigits) for bound in interval]
        lower, upper = f'({lower}', f'{upper}]'
        if interval == self.intervals[0]:
            lower = '(-inf'
        elif interval == self.intervals[-1]:
            upper = 'inf)'
        return f'{lower}; {upper}'

    def optimize(self, alpha=0.1, beta=0.001, gamma=5, max_iter=100, verbose=False):
        """
        Main method for iterating the dynamic discretization
        algorithm.

        Params
        ------
        alpha: float (optional)
            Stable-entropy-error stopping criteria.

        beta: float (optional)
            Low-entropy-error stopping criteria.

        gamma: float (optional)
            No-improvement stopping criteria.

        max_iter: int (optional)
            Maximum number of optimization interations.

        verbose: bool (optional)
            Verbose outputs flag.

        Returns
        ------
        self: DynamicDiscretizer
            Returns self for chained method calling.
        """
        err = np.sum([self._compute_error(x) for x in self.intervals])
        self.errors.append(err)

        for i in range(max_iter):
            self._split()
            self._merge(beta/5)
            err = np.sum([self._compute_error(x) for x in self.intervals])
            self.errors.append(err)

            if len(self.errors) > 3:
                last3 = np.array(self.errors[-4: -1]) / np.array(self.errors[-3:])
                stable = np.logical_and(last3 >= (1-alpha), last3 <= (1+alpha))
                if all(stable):
                    if verbose:
                        print('Stable-entropy-error stopping rule')
                    break

            if err < beta:
                if verbose:
                    print('Low-entropy-error stopping rule')
                break

            if (len(self.errors) - np.argmin(self.errors)) >= gamma:
                if verbose:
                    print(f'No-improvement stopping rule')
                break

        return self

    def visualize(self):
        """
        Plotting method for visualizing the initial and
        final dynamic discretization with KDE overlay.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6, 3], sharey=True)
        sns.histplot(self.X, kde=True, stat='density',
                    bins=self.initial, ax=ax1)
        sns.histplot(self.X, kde=True, stat='density',
                    bins=self.edges, color='g', ax=ax2)

        opts = dict(
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=11
        )

        ax1.set(title='Initial Static Discretization')
        ax1.text(0.98, 0.97, f'Error = {round(self.errors[0], 4)}',
                **opts, transform=ax1.transAxes)
        ax1.text(0.98, 0.89, f'N-Intervals = {len(self.initial) - 1}',
                **opts, transform=ax1.transAxes)

        ax2.set(title='Final Dynamic Discretization')
        ax2.text(0.98, 0.97, f'Error = {round(self.errors[-1], 4)}',
                **opts, transform=ax2.transAxes)
        ax2.text(0.98, 0.89, f'N-Intervals = {len(self.intervals)}',
                **opts, transform=ax2.transAxes)

        plt.tight_layout()
        # plt.show()
        plt.savefig('top_half.png', dpi=300, bbox_inches='tight')

    def draw_error(self):
        """
        Plotting method for visualizing the relative entropy
        error across all iterations.
        """
        fig, ax = plt.subplots(figsize=[6.4, 1.5])
        ax.semilogy(self.errors)
        ax.set(
            xlabel='Iterations',
            ylabel='Relative Entropy Error'
        )


class StaticDiscretizer(Discretizer):
    """
    Static Discretization class for Bayesian networks.

    Discretizes continuous variables based on strategies
    and intervals available to the SKLearn KBinsDiscretizer
    class.
    """
    def __init__(self, X, bins=4, strategy='uniform'):
        """
        Params
        ------
        X: pandas.Series
            The feature you wish to discretize.

        bins: int (optional)
            The target number of discrete intervals.

        strategy: string (optional)
            One of 'uniform' | 'quantile' | 'kmeans'.
        """
        super().__init__()

        self.disc = KBinsDiscretizer(bins, encode='ordinal', strategy=strategy)
        self.disc.fit(X.to_frame())
        self.edges = self.disc.bin_edges_[0]

        self.X = X.copy()

        self._compute_intervals()

    def _stringify(self, interval, ndigits=3):
        """
        Convenience method for representing an interval using
        mathematical notation with a closed lower bound and
        open upper bound.

        Params
        ------
        interval: tuple
            Interval edges of shape (2,).

        ndigits: int (optional)
            Number of significant digits for rounding values.

        Returns
        -------
        interval: string
            Interval bounds in mathematical notation.
        """
        lower, upper = [round(bound, ndigits) for bound in interval]
        lower, upper = f'({lower}', f'{upper}]'
        if interval == self.intervals[0]:
            lower = '(-inf'
        elif interval == self.intervals[-1]:
            upper = 'inf)'
        return f'{lower}; {upper}'


def prepare_static(df, bins=5, strategy='uniform'):
    """
    Apply static discretization to a complete dataset.

    Params
    ------
    df: pandas.DataFrame
        The dataset you wish to discretize.

    bins: int (optional)
        The target number of discrete intervals.

    strategy: string (optional)
        One of 'uniform' | 'quantile' | 'kmeans'.

    Returns
    -------
    trn: pandas.DataFrame
        Discretized dataset with all columns as string dtype.
    """
    trn = df.copy()
    for col in trn:
        if trn[col].dtype in ['int', 'float']:
            disc = StaticDiscretizer(trn[col], bins=bins, strategy=strategy)
            trn[col] = disc.transform()

    return trn.astype(str)


def prepare_dynamic(df, bins=5, verbose=False, disc_kws={}):
    """
    Apply dynamic discretization to a complete dataset.

    Params
    ------
    df: pandas.DataFrame
        The dataset you wish to discretize.

    bins: int (optional)
        The number of initial uniform discrete intervals.

    verbose: bool (optional)
        Verbose outputs flag.

    disc_kws: dict (optional)
        Keyword arguments to pass to DynamicDiscretizer.optimize.

    Returns
    -------
    trn: pandas.DataFrame
        Discretized dataset with all columns as string dtype.
    """
    trn = df.copy()
    for col in trn:
        if trn[col].dtype in ['int', 'float']:
            disc = DynamicDiscretizer(trn[col], bins=bins)
            trn[col] = disc.optimize(**disc_kws).transform()
            if verbose:
                disc.visualize()

    return trn.astype(str)
