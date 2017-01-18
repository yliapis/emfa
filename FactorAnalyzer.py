import numpy as np


class FactorAnalyzer_old:
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
        XX = X[:,:,None] # n x p x 1 , for nasty tensor stuff...
        LL = [] # log likelihood history
        T = (0,2,1) # transpose order
        for i in range(iters):
            ### E - step
            Psi_i = (1/np.diag(Psi))*mask # easy inverse
            XXT = XX.transpose(T)
            beta = Lambda.T@(Psi_i - Psi_i@Lambda@\
                             inv(I + Lambda.T@Psi_i@Lambda)@\
                             Lambda.T@Psi_i) # k x p
            E_zx = beta @ XX #n x k x 1
            E_zxT = E_zx.transpose(T)
            E_zzx = I - beta@Lambda + \
                (beta @ (XX @ XXT) @ beta.T) # n x k x k
            ### Compute Log Likelihood
            ll = -(n/2)*np.linalg.det(Psi) - \
                ((.5*XXT@Psi_i@XX -XXT@Psi_i@Lambda@E_zx).squeeze() + \
                np.trace(Lambda.T@Psi_i@Lambda@E_zzx, \
                         axis1=1, axis2=2)).sum(0)
#             print(ll.shape)
#             pshape = lambda A: print(A.shape)
#             pshape(Lambda.T@Psi_i@Lambda@E_zzx)
#             pshape(np.trace(Lambda.T@Psi_i@Lambda@E_zzx, \
#                          axis1=1, axis2=2))
#             pshape(.5*XXT@Psi_i@XX -XXT@Psi_i@Lambda@E_zx)
            LL.append(ll)
            ### M - step
            Lambda = (XX@E_zxT).sum(0) @ inv(E_zzx.sum(0))
            Psi = mask * (XX@XXT - \
                          Lambda@E_zx@XXT).mean(0)
            ###
            if verbose and (i+1)%10==0:
                print("cycle {}, log-likelihood {}".format(i+1, ll))
        ###
        self.Lambda = Lambda
        self.Psi = Psi
        self.LL = LL
        return Lambda, Psi


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

        