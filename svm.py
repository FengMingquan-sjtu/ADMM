# admm version of linear svm
# reference:
# https://web.stanford.edu/~boyd/papers/admm/svm/linear_svm_example.html
# https://web.stanford.edu/~boyd/papers/admm/svm/linear_svm.html
# Section 8.2.3 in 2011-boyd-Distributed optimization and statistical learning via the alternating direction method of multipliers
import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


import utils

class ADMM_SVM:
    def __init__(self, lamb=1, rho=1, n_jobs=2, max_iter=1000):
        self.lamb = lamb
        self.rho = rho
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        

    def fit(self, X, y):
        if y.min()==0:
            print("convert 0/1 label to -1/1")
            y = (y-0.5)*2 #convert 0/1 label to -1/1.
        n_sample, dim = X.shape
        # X is input data(n_sample * dim), y is label (n_sample).
        # A = - y .* [X, 1],  shape = (n_sample * (dim+1))
        A = - np.hstack((X, np.ones((n_sample,1)))) * np.expand_dims(y, axis=1)
        self.A = A
       
        # partition of A: randomly assign n_sample to n_jobs cores.
        self.p = np.random.randint(low=0, high=self.n_jobs, size=(n_sample,))
        
        #extreme partition
        #self.p = np.zeros(n_sample)
        #self.p[y==-1] = np.random.randint(low=0, high=self.n_jobs//2, size=((y==1).sum(),))
        #self.p[y== 1] = np.random.randint(low=self.n_jobs//2, high=self.n_jobs, size=((y==-1).sum(),))
        

        # optimization variables
        self.x = np.random.rand(dim+1, self.n_jobs) # x is optimization variable (w,b). 
        self.z = np.random.rand(dim+1, 1)
        self.u = np.random.rand(dim+1, self.n_jobs)
        #log
        log = dict()
        
        # stop criteria
        not_update_max_iter = 200
        not_update_n_iter = 0

        for i in range(self.max_iter):
            arg_list = [(j,) for j in range(self.n_jobs)]
            with Pool(self.n_jobs) as p:
                #x_list = p.starmap(self._update_x_sgd, arg_list)
                x_list = p.starmap(self._update_x_cvx, arg_list)
            
            self.x = np.array(x_list).T
            #print(x.shape)
            #self.z = 1/(1/(self.lamb*self.rho)+ self.n_jobs) * (self.x+self.u).mean(axis=1,keepdims=True)
            self.z = self._update_z_cvx()
            self.u += self.x- self.z
            
            self._update_log(log)
            print("---iter  %d----"%i)
            print("log[obj] = ", log["obj"][-1])
            print("log[constr] = ", log["constr"][-1])
            print("score = ", self.score(X,y) )

            if log["obj"][-1] <= min(log["obj"]):
                not_update_n_iter = 0
            else:
                not_update_n_iter += 1
                if not_update_n_iter > not_update_max_iter:
                    break
        utils.draw(log)

    def _update_x_sgd(self, j):
        # update x[j]
        # minimize ( sum(pos(A{i}*x_var + 1)) + rho/2*sum_square(x_var - z(:,i) + u(:,i)) )
        # f=lambda x_j:  F.relu(A[p==j].matmul(x_j) + 1 ).sum() + self.rho/2 *(x_j - z + u[:,j]).square().sum()

        batch_size = 32
        max_iter_inner = 1000
        lr = 0.001
        
        not_update_max_iter = 20
        not_update_n_iter = 0
        min_loss = 1e20

        x = torch.tensor(self.x[:,j]).double().requires_grad_(True)
        A_j = torch.tensor(self.A[self.p==j]).double()
        u_j = torch.tensor(self.u[:,j]).double()
        z_j = torch.tensor(self.z[:,0]).double()
        
        optimizer = torch.optim.Adam([x],lr=lr)

        for i in range(max_iter_inner):
            idx = torch.randperm(A_j.shape[0])
            losses = []
            for b in range(len(idx) // batch_size):
                batch_idx = idx[b*batch_size: (b+1)*batch_size]
                optimizer.zero_grad()
                loss = self._loss_sgd(A_j, x, z_j, u_j)
                loss.backward()
                losses.append(loss.detach())
                optimizer.step()
            avg_loss = sum(losses)/len(losses)
            #print("%s: inner iter %d, loss=%.4f"%(name, i, avg_loss))
            if avg_loss < min_loss:
                not_update_n_iter = 0
                min_loss = avg_loss
            else:
                not_update_n_iter += 1
                if not_update_n_iter > not_update_max_iter:
                    #print("%s: inner iter stops"%(name,))
                    break
        return x.detach().numpy() 


    
    def _loss_sgd(self, A_j, x_j, z_j, u_j):
        return F.relu(A_j.matmul(x_j) + 1).sum() + self.rho/2 *(x_j - z_j + u_j).square().sum()

    def _update_x_cvx(self, j):
        #minimize ( sum(pos(A{i}*x_var + 1)) + rho/2*sum_square(x_var - z(:,i) + u(:,i)) )
        
        x = cp.Variable(self.x.shape[0])
        objective = cp.Minimize(cp.sum(cp.pos(self.A[self.p==j] @ x + 1)) + self.rho/2 * cp.sum_squares(x - self.z[:,0] + self.u[:,j]))
        cp.Problem(objective).solve(verbose=False, solver=cp.SCS)
        return x.value

    def _update_z_cvx(self):
        z = cp.Variable(self.z.shape)
        objective = cp.Minimize(cp.sum_squares(z[:-1]) + self.rho/2 * cp.sum_squares(self.x - z + self.u) )
        cp.Problem(objective).solve()
        return np.array(z.value)

    def _objective(self, A, p, x, z):
        #obj = hinge_loss(A,x) + 1/(2*lambda)*sum_square(z(:,1));
        obj = 0
        for j in range(self.n_jobs):
            obj += np.maximum(np.matmul(A[p==j],x[:,j]) + 1, np.zeros(A[p==j].shape[0])).sum()
        obj += 1/(2 * self.lamb) * np.square(z[:-1]).sum()
        return obj
    
    def _regularization(self, z):
        return np.square(z[:-1]).sum()

    def _constraint(self, x, z):
        return np.square(x-z).sum()

    def _update_log(self, log):
        if len(log) == 0: #initialization
            log["obj"] = []
            log["constr"] = []
            log["regular"]= []
        log["obj"].append(self._objective(self.A, self.p, self.x, self.z))
        log["constr"].append(self._constraint(self.x, self.z))
        log["regular"].append(self._regularization(self.z))


    def score(self, X, y):
        X = np.hstack((X, np.ones((X.shape[0],1))))
        y_hat = np.matmul(X, self.x[:,0]) > 0 
        y_hat = (y_hat - 0.5) * 2
        if y.min()==0:
            #print("convert 0/1 label to -1/1")
            y = (y-0.5)*2 
        score = (y_hat==y).sum()/len(y)
        return score

def get_synthetic_data():
    n_pos_sample, n_neg_sample = 100, 100
    pos_sample = [[1.5 + 0.9*np.random.rand(1, int(n_pos_sample*0.6)),  1.5 + 0.7*np.random.rand(1, int(n_pos_sample*0.4))],
                  [2   + 2  *np.random.rand(1, int(n_pos_sample*0.6)),  -2  + 2  *np.random.rand(1, int(n_pos_sample*0.4))] ]
    pos_sample = np.vstack((np.hstack(pos_sample[0]), 
                            np.hstack(pos_sample[1])))
    neg_sample = [[-1.5 + 0.9*np.random.rand(1, int(n_neg_sample*0.6)),  -1.5 + 0.7*np.random.rand(1, int(n_neg_sample*0.4))],
                  [-2   + 2  *np.random.rand(1, int(n_neg_sample*0.6)),  2    + 2  *np.random.rand(1, int(n_neg_sample*0.4))]]
    neg_sample = np.vstack((np.hstack(neg_sample[0]), 
                            np.hstack(neg_sample[1])))
    X = np.hstack((pos_sample, neg_sample)).T

    y = np.zeros(n_pos_sample + n_neg_sample)
    y[:n_pos_sample] = 1
    # X.shape= (200,2),  y.shape =(200,)
    return X,y

def test_on_fashion():
    model = ADMM_SVM(max_iter=1000, n_jobs=20)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=8), model)
    X, y = np.load("../ml-hw-1/data/X_test_hog.npy"), np.load("../ml-hw-1/data/y_test_sampled.npy")
    n_sample = 1000
    idx_ = np.random.rand(y.shape[0]) < n_sample/len(y)
    X, y = X[idx_], y[idx_]
    pipe.fit(X,y)

def test_on_synthetic():
    model = ADMM_SVM(max_iter=400, n_jobs=20)
    X,y = get_synthetic_data()
    model.fit(X, y)

if __name__ == "__main__":
    #test_on_synthetic()
    test_on_fashion()
        