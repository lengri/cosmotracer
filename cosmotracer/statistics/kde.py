import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

class GaussianKDE:
    
    """
    This is a pretty inefficient class to calculate KDEs using
    variable bandwidths and weighted kernels/samples.
    
    Scipy only allows weighting, but requires fixed bandwidth, so
    for now we're stuck with this class.
    """
    
    def __init__(
        self,
        mu : np.ndarray,
        error : np.ndarray,
        weights : np.ndarray | None = None,
        x : np.ndarray | None = None,
        allow_negative : bool = False        
    ):
        self.mu = mu 
        self.error = error 
        self.weights = (weights/np.sum(weights)) 
        
        # check if x has been defined. If not, construct it from 
        # input dataset...
        if x is None:
            
            min_val = self.mu.min()-self.error.max()*5
            # NOTE: This will allow negative values if allow_negative is true
            # The problem is of course that Gaussian Kernels will always 
            # be implicitly defined for x < 0 and thus we will "bleed" 
            # some probability into the negative values. Solution would be
            # to change the kernel, or be very careful with the bandwidths?
            if not allow_negative and min_val < 0.:
                min_val = 0.
            self.x = np.linspace(
                min_val, 
                self.mu.max()+self.error.max()*5,
                500
            )
            
        else:
            # explicitly setting x ignores allow_negative for now.
            self.x = x 
            
        # construct the KDE...
        self.ykde = self.evaluate_kde(
            self.x,
            exact=True
        )
        # Calling the kde is handled by the kde method.
        
        # construct a cdf by calculating integral over KDE...
        self.ycdf = sp.integrate.cumulative_trapezoid(
            y=self.ykde, x=self.x, initial=0
        )

        # Also construct the inverse CDF that returns the x corresponding
        # to a defined probability. 
        self.x_invcdf = self.ycdf.copy()
        self.y_invcdf = self.x.copy()
        
    def _gaussian(
        self, 
        x, 
        m, 
        s
    ):
        pdf = (1/(np.sqrt(2*np.pi)*s))*np.exp(-((x-m)**2)/(2*s**2))
        return pdf
       
    def _interp_kde(self, x):
        return np.interp(
            x, self.x, self.ykde, left=0., right=0.
        )
    
    def _interp_cdf(self, x):
        return np.interp(
            x, self.x, self.ycdf, left=0., right=1.
        )
        
    def _interp_invcdf(self, x):
        return np.interp(
            x, self.x_invcdf, self.y_invcdf, left=np.nan, right=np.nan
        )
    
    def evaluate_kde(
        self,
        x : float | np.ndarray,
        exact : bool = False
    ):
        if isinstance(x, np.ndarray):
            yout = np.zeros_like(x, dtype=float)
        else:
            yout = 0.
            
        if exact:
            for (m, s, w) in zip(self.mu, self.error, self.weights):
                yout += w*self._gaussian(x, m, s)
        else:
            yout = self._interp_kde(x) # interpolate the value from precomputed KDE
        
        return yout
    
    def evaluate_cdf(
        self,
        x,
    ):
        yout = self._interp_cdf(x)
        return yout
    
    def evaluate_invcdf(
        self,
        x
    ):
        yout = self._interp_invcdf(x)
        return yout
        
    def sample_kde(
        self,
        n,
        dx : float | None = None
    ):
        if dx is not None:
            xsample = np.arange(self.x.min(), self.x.max()+dx, dx)
        else:
            xsample = self.x 
        
        p = self.evaluate_kde(xsample)
        p /= np.sum(p)
        
        sample = np.random.choice(
            a=xsample,
            size=n,
            replace=True,
            p=p
        )
        return sample
    

if __name__ == "__main__":
    mu = np.random.uniform(low=25, high=200, size=500)
    error = np.random.uniform(10, 40, size=len(mu))
    weights = np.random.uniform(0.1, 10, size=len(mu))
    
    gk = GaussianKDE(mu, error, weights)
    x = np.linspace(0, 250, 500)
    
    sample = gk.sample_kde(n=1000)
    
    fg, ax = plt.subplots(1, 2)
    ax[0].hist(sample, density=True)
    ax[0].plot(x, gk.evaluate_kde(x, exact=True))
    ax[0].plot(x, gk.evaluate_kde(x, exact=False))
    
    ax[1].plot(x, gk.evaluate_cdf(x))
    print(gk.evaluate_invcdf(np.array([0., 0.001, 0.25, 0.5, 0.75, 1.])))
    plt.show()