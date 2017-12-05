import numpy as np
import pandas as pd
import itertools
import peakutils
from scipy.stats.kde import gaussian_kde

__all__ = ['Bootlier', 'boot', 'Hratio', 'find_hratio', 'find_outliers']


def _part(n, k):
    """Integer partitioning
       from https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions
    """
    def _kpart(n, k, pre):
        if n <= 0:
            return []
        if k == 1:
            if n <= pre:
                return [[n]]
            return []
        ret = []
        for i in range(min(pre, n), 0, -1):
            ret += [[i] + sub for sub in _kpart(n-i, k-1, i)]
        return ret
    return _kpart(n, k, n)


class Bootlier(object):
    """Samples and MTM for a list of points for bootlier."""

    def __init__(self, npoints, z, b, k):
        df = self._make_samples(npoints, z, b)
        df = self._calc_means(df)
        df = self._calc_trimmed_means(df, k)
        self.samples = df

    def _make_samples(self, npoints, z, b):
        samples = pd.DataFrame(columns=['sample'])
        for i in range(b):
            sample = [np.random.choice(npoints, size=z, replace=True)]
            samples.loc[i] = sample
        return samples

    def _calc_means(self, samples):
        samples['mean'] = samples['sample'].apply(lambda x: np.mean(x))
        return samples

    def _calc_trimmed_means(self, samples, k):
        samples['sample'] = samples['sample'].apply(lambda x: np.sort(x))
        samples['trimmed_mean'] = samples['sample'].apply(lambda x:
                                                          np.mean(x[k:-k]))
        samples['mtm'] = (samples['mean'].values -
                          samples['trimmed_mean'].values)
        return samples


def boot(npoints, z=None, b=500, k=2):
    """
    Parameters
    ----------
    npoints : `list`
        List of N points from which to draw samples.
    z : `int`
        Number of points in each bootstrapped sample.
    b : `int`
        Number of bootstraps to draw.
    k : `int`
        Number of points to trim from each extreme side for
        the trimmed mean.
    """
    if z is None:
        z = len(npoints)
    samples = Bootlier(npoints, z, b, k)
    return samples.samples


class Hratio(object):

    def __init__(self, mtmlist):
        mtmrange = max(mtmlist) - min(mtmlist)
        x = np.arange(min(mtmlist), max(mtmlist), mtmrange/100.)

        try:
            kde_orig = gaussian_kde(mtmlist, bw_method='silverman')
            self.horig = kde_orig.factor
            hrange = np.arange(0.1*self.horig, 10*self.horig, 0.02*self.horig)
            self.horig_kde = kde_orig
            peakind = peakutils.indexes(kde_orig(x), thres=0.02/max(kde_orig(x)))
            self.horig_peak = [(x[peak], kde_orig(x)[peak]) for peak in peakind]
            self.numpeaks = len(peakind)

            i = 0
            peaks = 100
            while peaks > 1:
                hcrit = hrange[i]
                kde = gaussian_kde(mtmlist, bw_method=hcrit)
                peakind = peakutils.indexes(kde(x), thres=0.02/max(kde(x)))
                peaks = len(peakind)
                i += 1
            self.hcrit = hrange[i-1]
            kde = gaussian_kde(mtmlist, bw_method=hrange[i-1])
            peakind = peakutils.indexes(kde(x), thres=0.02/max(kde(x)))
            self.hcrit_peak = [(x[peak], kde(x)[peak]) for peak in peakind]
            self.hcrit_kde = kde
            self.hratio = self.horig/hcrit
            self.hratio
        except:
            self.hratio = 100
            self.numpeaks = "Unknown"


def find_hratio(mtmlist):
    """
    Parameters
    ----------
    mtmlist : `list`
        List of points for making KDEs.
    Returns
    -------
    hratio : Hratio
        The ratio of the original h value to the smallest value of h for
        which the KDE has only one peak and other parameters.
        Less than one contains outliers.
    """
    hratio = Hratio(mtmlist)
    return hratio


def find_outliers(origpoints, sensitivity=1., detrend=False):
    """Find outliers in a list using a given sensitivity parameter.
    Parameters
    ----------
     npoints : `list`
        List of points for which to find outliers.
    sensitivity : `float`
        Sensitivity threshold for the cutoff hratio.
        Default of 1. Less than 1 is less sensitive to outliers,
        greater than 1 is more sensitive.
    Returns
    -------
    outliers
    """

    if detrend is True:
        i1, i2 = itertools.tee(iter(origpoints))
        next(i2)
        lst = [y-x for x, y in zip(i1, i2)]
        #lst.insert(0, origpoints[0])
        npoints = lst
        points = sorted(npoints)
    else:
        points = sorted(origpoints)

    for i in range(0, int(len(points)/2)):
        if i != 0:
            a = points[0:-i]
            if len(a) > 1:
                boota = boot(a)
                ha = find_hratio(boota['mtm'])
                hrat = ha.hratio
                if hrat >= sensitivity:
                    remaining = a
                    break

        b = points[i:]
        if len(b) > 1:
            bootb = boot(b)
            hb = find_hratio(bootb['mtm'])
            hrat = hb.hratio
            if hrat >= sensitivity:
                remaining = b
                break

        p = _part(i, 2)

        for pair in p:
            a = points[pair[0]:-pair[1]]
            if len(a) > 1:
                boota = boot(a)
                ha = find_hratio(boota['mtm'])
                hrat = ha.hratio
                if hrat >= sensitivity:
                    remaining = a
                    break

            b = points[pair[1]:-pair[0]]
            if len(b) > 1:
                bootb = boot(b)
                hb = find_hratio(bootb['mtm'])
                hrat = hb.hratio
                if hrat >= sensitivity:
                    remaining = b
                    break
        if hrat >= sensitivity:
                break

    if detrend is True:
        outtrend = [x for x in npoints if x not in remaining]
        outindex = [npoints.index(x) for x in outtrend]
        outliers = [origpoints[i] for i in outindex]

        return outindex, outliers
    else:
        try:
            outliers = [x for x in origpoints if x not in remaining]
            outindex = [origpoints.tolist().index(x) for x in outliers]
        except AttributeError:
            outliers = [x for x in origpoints if x not in remaining]
            outindex = [origpoints.index(x) for x in outliers]

        return outindex, outliers
