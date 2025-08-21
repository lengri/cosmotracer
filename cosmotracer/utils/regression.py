import numpy as np

def linear_regression(x, y):

    """
    Basic equation for linear regression.
    """
    
    x = np.array(x)
    y = np.array(y)
    
    # X matrix
    X = np.ones((len(x), 2))
    X[:,1] = x
    
    # Solve matrix equations
    Xt = np.transpose(X)
    par = np.matmul(
        np.linalg.inv(np.matmul(Xt, X)),
        np.matmul(Xt, y)
    )
    
    # residuals
    ypred = par[0] + par[1]*x
    
    yhatdash = np.mean(ypred)
    ydash = np.mean(y)
    
    r2 = np.sum((ypred-yhatdash)**2) / np.sum((y-ydash)**2)
    
    return par, r2 