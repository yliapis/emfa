import numpy as np


class FactorAnalyzer:
    '''
        Implementing single factor analysis model from following paper
            "The EM algorithm for mixtures of factor analyzers", Ghahrmani
        
        Naming convensions are used as in paper
            k: number of factors
            p: number of observed variables
                        
    '''
    
    def __init__(self):
        pass
        
    def fit(self, data, n_factors=3, iters=100, verbose=False):
        '''
        data: (instance x p) dimensional numpy array
        iters: number of iterations to run EM algo
        
        Model:
            z = Lambda @ x + Psi
        
        returns:
            Lambda: factor loading matrix (p x k)
            Psi: diagonal matrix
        
        '''
        ### Preprocessing
        # store vars
        X = data
        n, p = X.shape
        k = n_factors
        # demean data
        X = X - np.ones((n,1)) * X.mean(0)
        ### Initialize model params: Lambda, Psi
        Lambda = np.random.rand(p,k)
        mask = np.eye(p)
        Psi = mask*(np.random.rand(p,p)+.001)
        
        # misc
        I = np.eye(k)
        inv = np.linalg.inv
        ### Run EM
        LL = [] # log likelihood history
        ll_max = 0
        for i in range(iters):
            ### E - step
            Psi_i = (1/np.diag(Psi))*mask # easy inverse
            beta = Lambda.T@(Psi_i - Psi_i@Lambda@\
                             inv(I + Lambda.T@Psi_i@Lambda)@\
                             Lambda.T@Psi_i) # k x p
            E_zx = beta @ X.T #k x n
            E_zzx = I - beta@Lambda + \
                (beta @ (X.T@X) @ beta.T) #k x k
            ### Compute Log Likelihood
            ll = (p/2)*np.log(2*np.pi)-(n/2)*np.linalg.det(Psi) - \
                ((.5*(X@Psi_i*X).sum() -(X@Psi_i@Lambda*E_zx.T).sum()) + \
                np.trace(Lambda.T@Psi_i@Lambda@E_zzx))
            if ll > ll_max:
                ll_max = ll
                self.Lambda = Lambda
                self.Psi = Psi
                self.i = i
            LL.append(ll)
            ### M - step
            Lambda = (X.T@E_zx.T) @ inv(E_zzx)
            Psi = mask * (X.T@X - \
                          Lambda@E_zx@X)/n
            ###
            if verbose:                
                print("cycle {}, log-likelihood {}".format(i+1, ll))
        ###
        self.LL = LL
        return self.Lambda, self.Psi

