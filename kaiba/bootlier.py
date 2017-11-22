import numpy as np
import pandas as pd

__all__ = ['Bootlier']


class Bootlier(object):
    """Calculates the M-TM for a list of points for bootlier."""

    def __init__(self):

        pass

    def _make_samples(self, npoints, b, z):
        samples = pd.DataFrame(columns=['sample'])
        for i in range(z):
            sample = [np.random.choice(npoints, size=b, replace=True)]
            samples.loc[i] = sample
        return samples

    def _calc_means(self, samples):
        samples['mean'] = samples['sample'].apply(lambda x: np.mean(x))
        return samples

    def _calc_trimmed_means(self, samples, k):
        samples['sample'] = samples['sample'].apply(lambda x: np.sort(x))
        samples['trimmed_mean'] = samples['sample'].apply(lambda x: np.mean(x[k:-k]))
        samples['m_tm'] = samples['mean'].values - samples['trimmed_mean'].values
        return samples

    def boot(self, npoints, b, z, k):
        """
        Parameters
        ----------
        npoints : `list`
            List of N points from which to draw samples.
        b : `int`
            Number of points in each bootstrapped sample.
        z : `int`
            Number of bootstraps to draw.
        k : `int`
            Number of points to trim from each extreme side for the trimmed mean.
        """
        df = self._make_samples(npoints, b, z)
        df = self._calc_means(df)
        df = self._calc_trimmed_means(df, k)
        return df
