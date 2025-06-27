import numpy as np
import riserfit as rf 
from numba import njit
from riserfit import (
    DistributionFromInterpolator,
    distribution_from_sample
)
import scipy as sp

class GradientFactory():
    def __init__(self, shape):
        self.shape = shape
        
    def create_uniform(
        self,
        value
    ):
        return np.full(self.shape, value)

    def create_linear(
        self,
        value_pair,
    ):
        
        y0 = value_pair[0]
        y1 = value_pair[1]
        
        m = self.shape[1]
        e = (y1-y0)/(m-1)*np.arange(0,m,1)+y0
        
        return np.full(self.shape, e)

    def create_step(
        self,
        value_pair
    ):
        y0 = value_pair[0]
        y1 = value_pair[1]
        
        e = np.ones(self.shape)
        e[:,:int(self.shape[1]/2)] = y0
        e[:,int(self.shape[1]/2):] = y1 
        
        return e
    
def _is_numeric(value):
    if type(value) == np.ndarray: return False 
    try:
        float(value)
        return True
    except:
        return False


@njit
def gaussian_kernel_numba(x, mu, sigma):
    frac = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -((x - mu)**2) / (2.0 * sigma**2)
    return frac * np.exp(exponent)

@njit
def kde_gaussian(x, xi, hi, wi):
    
    tot_sum = np.zeros(x.shape[0])
    for i in range(xi.shape[0]):
        mu = xi[i]
        h = hi[i]
        w = wi[i]
        tot_sum += w * gaussian_kernel_numba(x, mu, h)
    area = np.trapz(tot_sum, x)
    kde = tot_sum / area
    return kde
    
class SyntheticDistribution():
    
    def __init__(self, xi, dx, h_rule = lambda x : x*0.05):
        self.dx = dx
        self.h_rule = h_rule
        hi = h_rule(xi)
        self.min = max(xi.min()-hi.max()*5, 0)
        self.max = xi.max()+hi.max()*5
        self.x = np.arange(self.min, self.max+dx, dx)

        self.y = kde_gaussian(
            self.x,
            xi=xi,
            hi=hi,
            wi=np.ones(xi.shape)
        )
        
        self.pdf = sp.interpolate.interp1d(
            self.x, self.y, bounds_error=False,
            fill_value=(0, 0)
        )
    
    def draw_sample(self, size):
        
        self.sample = np.random.choice(
            self.x,
            size=size,
            replace=True,
            p = self.y/np.sum(self.y) 
        )
        return self.sample

    def calculate_sample_pdf(self):
        hi = self.h_rule(self.sample)
        self.sample_y = kde_gaussian(
            self.x,
            xi=self.sample,
            hi=hi,
            wi=np.ones(self.sample.shape)
        )
        self.sample_pdf = sp.interpolate.interp1d(
            self.x, self.sample_y, bounds_error=False,
            fill_value=(0, 0)
        )
    
    def calculate_sample_misfit(self):
        # calculate the sample kde
        self.mf = np.trapz(
            np.abs(self.y-self.sample_y),
            x=self.x
        )
        return self.mf
    
    def repeat_n_draws(self, n, size):
        
        self.repeat_mfs = np.zeros(n)
        
        for i in range(0, n):
            self.draw_sample(size=size)
            self.calculate_sample_pdf()
            self.repeat_mfs[i] = self.calculate_sample_misfit()
        
        return self.repeat_mfs
        

if __name__ == "__main__":

    vals = np.random.normal(1000, 50, 5000)
    
    dist = distribution_from_sample(
        vals, 10
    )
    
    sd = SyntheticDistribution(vals, 10)
    sd.draw_sample(200)
    sd.repeat_n_draws(n=10, size=200)
    print(sd.repeat_mfs)
    
    import matplotlib.pyplot as plt
    plt.plot(sd.x, sd.y)
    plt.plot(sd.x, sd.sample_y)
    plt.plot(dist.x, dist.density)
    plt.show()
