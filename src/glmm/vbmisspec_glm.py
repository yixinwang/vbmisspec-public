# python 2
# stan-helper 0.8

import numpy as np
import numpy.random as npr
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=60)

import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib import style
style.use('ggplot')
import itertools

from IPython.display import display

import stanhelper
import seaborn as sns
# from functools import reduce
import subprocess
import math
from scipy import stats



K = 4
alpha = -0.5
sigma = 2
normal_mean = [0, 0] #for first two dims of x
normal_cov = [[1, 0], [0, 25]]
bin_p = [0.4, 0.8] #for last two dims of x
beta = [0.2, -0.2, 2, -2]
gp = 10

r = 1
test_M = 1000
test_N = test_M * gp
test_x = np.zeros(shape = (test_N, K))
test_x[:,0:2] = np.random.multivariate_normal(normal_mean, normal_cov, test_N)
test_x[:, 2:4] = [np.random.binomial(1, p = bin_p) for i in range(test_N)]
test_g = [val for val in range(test_M) for _ in range(gp)]
test_a = np.random.normal(loc = 0, scale = sigma, size=test_M)
test_log_m = alpha + test_a[test_g].T + np.matmul(test_x, beta)
test_m = np.array([math.exp(x) for x in test_log_m])
test_g = [(val+1) for val in range(test_M) for _ in range(gp)]
test_p = (test_m / r) / (1 + (test_m / r))
# test_y = np.random.poisson(lam = test_m)
test_y = npr.negative_binomial(r, test_p)


# In[26]:


def plmm(size, algo):
    M = size
    N = M * gp
    train_true = {}
    train_true['K'] = K
    train_true['M'] = M
    train_true['N'] = N
    train_true['alpha'] = alpha
    train_true['sigma'] = sigma
    train_true['beta'] = beta
    train_true['g'] = [val for val in range(M) for _ in range(gp)]
    train_true['normal_mean'] = normal_mean
    train_true['normal_cov'] = normal_cov
    train_true['bin_p'] = bin_p
    train_true['x'] = np.zeros(shape = (N, K))
    train_true['x'][:, 0:2] = np.random.multivariate_normal(normal_mean,                                                             normal_cov, N)
    train_true['x'][:, 2:4] = [np.random.binomial(1, p = bin_p)                                for i in range(N)]
    train_true['a'] = np.random.normal(loc = 0, scale = sigma, size=M)
    train_true['log_m'] = (train_true['alpha'] +                                 train_true['a'][train_true['g']].T +                                 np.matmul(train_true['x'], train_true['beta']))
    train_true['m'] = np.array([math.exp(x) for x in train_true['log_m']])
#     train_true['y'] = np.random.poisson(lam = train_true['m'])
    train_true['p'] = (train_true['m'] / r) / (1 + (train_true['m'] / r))
    train_true['y'] = npr.negative_binomial(r, train_true['p'])
    train_true['g'] = [(val+1) for val in range(M) for _ in range(gp)] #reset starting from 0 for stan
    
    train_true['test_M'] = test_M
    train_true['test_N'] = test_N
    train_true['test_x'] = test_x
    train_true['test_a'] = np.random.normal(loc = 0, scale = sigma, size=test_M)
    train_true['test_log_m'] = test_log_m
    train_true['test_m'] = test_m
    train_true['test_g'] = test_g
    train_true['test_p'] = test_p
    train_true['test_y'] = test_y
    train_true_file = 'plmm_'+algo+'_M'+str(M)+'_train_true.data.R'
    stanhelper.stan_rdump(train_true, train_true_file)
    train_dict = {}
    train_dict['M'] = M
    train_dict['N'] = N
    train_dict['K'] = K
    train_dict['y'] = train_true['y']
    train_dict['x'] = train_true['x']
    train_dict['g'] = train_true['g']
    train_dict['test_M'] = train_true['test_M']
    train_dict['test_N'] = train_true['test_N']
    train_dict['test_y'] = train_true['test_y']
    train_dict['test_x'] = train_true['test_x']
    train_dict['test_g'] = train_true['test_g']    
    train_dict['test_a'] = train_true['test_a']    
    train_dict_file = 'plmm_'+algo+'_M'+str(M)+'_train_dict.data.R'
    stanhelper.stan_rdump(train_dict, train_dict_file)
    train_dict_init = {}
    train_dict_init['alpha'] = alpha
    train_dict_init['sigma'] = sigma
    train_dict_init['a'] = np.random.normal(loc = 0, scale = 1, size=M)
    train_dict_init['beta'] = beta
    train_dict_init_file = 'plmm_'+algo+'_M'+str(M)+'_train_dict_init.data.R'
    stanhelper.stan_rdump(train_dict_init, train_dict_init_file)
    output_file = 'plmm_'+algo+'_M'+str(M)+'_output.csv'
    subprocess.call('./plmm variational algorithm='+algo+' adapt                      iter=50000 engaged=0 eta=1.0 tol_rel_obj=1e-5                      data file='+train_dict_file                    +' output file='+output_file, shell=True)
    result = stanhelper.stan_read_csv(output_file)
    print(result['mean_pars'])  


# In[27]:


def plmm_sample(size, algo):
    M = size
    N = M * gp
    train_true = {}
    train_true['K'] = K
    train_true['M'] = M
    train_true['N'] = N
    train_true['alpha'] = alpha
    train_true['sigma'] = sigma
    train_true['beta'] = beta
    train_true['g'] = [val for val in range(M) for _ in range(gp)]
    train_true['normal_mean'] = normal_mean
    train_true['normal_cov'] = normal_cov
    train_true['bin_p'] = bin_p
    train_true['x'] = np.zeros(shape = (N, K))
    train_true['x'][:, 0:2] = np.random.multivariate_normal(normal_mean,                                                             normal_cov, N)
    train_true['x'][:, 2:4] = [np.random.binomial(1, p = bin_p)                                for i in range(N)]
    train_true['a'] = np.random.normal(loc = 0, scale = sigma, size=M)
    train_true['log_m'] = (train_true['alpha'] +                                 train_true['a'][train_true['g']].T +                                 np.matmul(train_true['x'], train_true['beta']))
    train_true['m'] = np.array([math.exp(x) for x in train_true['log_m']])
#     train_true['y'] = np.random.poisson(lam = train_true['m'])
    train_true['p'] = (train_true['m'] / r) / (1 + (train_true['m'] / r))
    train_true['y'] = npr.negative_binomial(r, train_true['p'])
    train_true['g'] = [(val+1) for val in range(M) for _ in range(gp)] #reset starting from 0 for stan
    
    train_true['test_M'] = test_M
    train_true['test_N'] = test_N
    train_true['test_x'] = test_x
    train_true['test_a'] = test_a
    train_true['test_log_m'] = test_log_m
    train_true['test_m'] = test_m
    train_true['test_g'] = test_g
    train_true['test_p'] = test_p
    train_true['test_y'] = test_y
    train_true_file = 'plmm_'+algo+'_M'+str(M)+'_train_true.data.R'
    stanhelper.stan_rdump(train_true, train_true_file)
    train_dict = {}
    train_dict['M'] = M
    train_dict['N'] = N
    train_dict['K'] = K
    train_dict['y'] = train_true['y']
    train_dict['x'] = train_true['x']
    train_dict['g'] = train_true['g']
    train_dict['test_M'] = train_true['test_M']
    train_dict['test_N'] = train_true['test_N']
    train_dict['test_y'] = train_true['test_y']
    train_dict['test_x'] = train_true['test_x']
    train_dict['test_g'] = train_true['test_g']    
    train_dict['test_a'] = train_true['test_a']    
    train_dict_file = 'plmm_'+algo+'_M'+str(M)+'_train_dict.data.R'
    stanhelper.stan_rdump(train_dict, train_dict_file)
    output_file = 'plmm_'+algo+'_M'+str(M)+'_output.csv'
    subprocess.call('./plmm sample                     data file='+train_dict_file                    +' output file='+output_file, shell=True)
    result = stanhelper.stan_read_csv(output_file)


for i in [5,10,20,50,100,200,500,1000,2000]:
    plmm(i, 'meanfield')
    plmm(i, 'fullrank')
    plmm_sample(i, 'sample')

