import numpy as np

__all__ = ['Bootlier']


class Bootlier(object):
    """Calculates the M-TM for a list of points using bootlier.

    Parameters
    ----------
    npoints : `list`
        List of N points from which to draw samples.
    z : `int`
        Number of bootstraps to draw.
    b : `int`
        Number of points in each bootstrapped sample.
    k : `int`
        Number of points to trim from each extreme side for the trimmed mean.
    """


    def __init__(self, npoints, z, b, k):
        self.npoints = npoints
        self.z = z
        self.b = b
        self.k = k

    def boot(self):
        bootstrap = np.random.choice(self.npoints, size=self.b, replace=True)
        return bootstrap

