import numpy as np


class FactorAnalyzer:
    '''
        Implementing single factor analysis model from following paper
            "The EM algorithm for mixtures of factor analyzers", Ghahrmani
        
        Naming convensions are used as in paper
                                    
    '''
    
    def __init__(self, n_factors=3):
        self.n_factors = n_factors

        
    def fit(self, data, iters=100, tol=.001, verbose=False):
        '''
        data: (instance x p) dimensional numpy array
        iters: number of iterations to run EM algo
        
        Model:
            z = Lambda @ x + Psi
        
        returns:
            Lambda: factor loading matrix (p x k)
            Psi: diagonal matrix
        
        '''
        ### Initialization
        X = data
        # store vars
        n, p = X.shape
        k = self.n_factors
        # demean data
        X = X - np.ones((n,1)) * X.mean(0)
        # Initialize model params: Lambda, Psi
        cX = np.cov(X.T)
        scale = np.sqrt(np.linalg.det(cX)/k)
        Lambda = np.random.rand(p,k).astype(np.float64)*scale
        mask = np.eye(p)
        Psi = mask * cX.astype(np.float64)
        # misc
        I = np.eye(k)
        #epsilon = 10**-9
        #Epsilon = Psi * epsilon # avoid numerical intability in Psi_inverse
        inv = np.linalg.inv
        c = (p/2)*np.log(2*np.pi)
        LL = [] # log likelihood history
        ll, ll_old = 0, 0
        #precompute
        XX = X.T @ X 
        X2 = X**2
        ### Run EM
        for i in range(iters):
            ### E - step
            Psi_i = (1/(np.diag(Psi)))*mask # easy inverse
            PLL = (Psi_i - Psi_i@Lambda@\
                             inv(I + Lambda.T@Psi_i@Lambda)@\
                             Lambda.T@Psi_i)
            beta = Lambda.T@PLL # k x p
            #E_zx = beta @ X.T #k x n
            # multiply 2 terms by n to compensate for summation
            E_zzx = I*n - n*beta@Lambda + \
                (beta @ XX @ beta.T) #k x k
            ### Compute Log Likelihood
            ll_old = ll
            ll = c-(n/2)*np.linalg.det(Psi) - \
                ((.5*(X2@Psi_i).sum() -(X2@Psi_i@(Lambda*beta.T)).sum()) + \
                np.trace(Lambda.T@Psi_i@Lambda@E_zzx))
            
            LL.append(ll)
            ### M - step
            Lambda = (XX @ beta.T) @ inv(E_zzx) #
            Psi = mask * (XX - Lambda@beta@XX)/n
            ###
            if verbose:                
                print("cycle {}, log-likelihood {}".format(i+1, ll))
            if i <=2:
                base = ll
            elif (ll-base) < (1+tol)*(ll_old-base) and (ll-ll_old) < 0.0:
                print("\tbreaking, LL={}".format(ll))
                break
            elif np.isnan(ll):
                print("\tbreaking: NaN")
                break
        ###
        self.LL = LL
        self.Lambda = Lambda
        self.Psi = Psi
        #
        return Lambda, Psi

