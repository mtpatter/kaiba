import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats.kde import gaussian_kde

__all__ = ['Bootlier']


class Bootlier(object):
    """Calculates the MTM for a list of points for bootlier."""

    def __init__(self):

        pass

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

    def boot(self, npoints, z, b, k=2):
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
        df = self._make_samples(npoints, z, b)
        df = self._calc_means(df)
        df = self._calc_trimmed_means(df, k)
        self.samples = df
        return df

    def find_hratio(self):
        """
        Returns
        -------
        hratio : `float`
            The ratio of the original h value to the smallest value of h for
            which the KDE has only one peak. Less than one contains outliers.
        """
        mtmlist = self.samples['mtm']
        mtmrange = max(mtmlist) - min(mtmlist)
        x = np.arange(min(mtmlist), max(mtmlist), mtmrange/100.)
        widths = mtmrange * np.arange(0.01, 1, 0.05)  # widths between peaks

        kde_orig = gaussian_kde(mtmlist, bw_method='silverman')
        self.horig = kde_orig.factor
        hrange = np.arange(0.01*self.horig, 10*self.horig, 0.01*self.horig)
        self.horig_kde = kde_orig
        peakind = signal.find_peaks_cwt(-kde_orig(x), widths)
        self.horig_peak = [(x[peak], kde_orig(x)[peak]) for peak in peakind]
        self.numpeaks = len(peakind)

        peaks = 1
        i = -1
        while peaks == 1:
            hcrit = hrange[i]
            kde = gaussian_kde(mtmlist, bw_method=hcrit)
            peakind = signal.find_peaks_cwt(-kde(x), widths)
            peaks = len(peakind)
            hcrit = hrange[i+1]
            kde.set_bandwidth(hcrit)
            i -= 1
        self.hcrit = hcrit
        kde.set_bandwidth(hrange[i+2])
        peakind = signal.find_peaks_cwt(-kde(x), widths)
        self.hcrit_peak = [(x[peak], kde(x)[peak]) for peak in peakind]
        self.hcrit_kde = kde
        self.hratio = self.horig/hcrit
        return self.hratio
