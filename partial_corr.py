"""

Taken from: https://gist.github.com/fabianp/9396204419c7b638d38f#file-partial_corr-py

Partial Correlation in Python (clone of Matlab's partialcorr)

This uses the linear regression approach to compute the partial 
correlation (might be slow for a huge number of variables). The 
algorithm is detailed here:

    http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression

Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
the algorithm can be summarized as

    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
    2) calculate the residuals in Step #1
    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
    4) calculate the residuals in Step #3
    5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 

    The result is the partial correlation between X and Y while controlling for the effect of Z


Date: Nov 2014
Author: Fabian Pedregosa-Izquierdo, f@bianp.net
Testing: Valentina Borghesani, valentinaborghesani@gmail.com
"""

import numpy as np
from scipy import stats, linalg

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.


    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr



'''
## copied from: https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py
cvar = np.atleast_2d(C[y_covar].to_numpy())
beta_y = np.linalg.lstsq(cvar, C[y].to_numpy(), rcond=None)[0]
res_y = C[y].to_numpy() - cvar @ beta_y


## reconfirm with matlab on complex case
## simple case was already confirmed, as below


#cvar = np.atleast_2d(C[y_covar].to_numpy())
cvar = controls44; # (29696, 18)
yvar =  grp_conn_tmpl_44;  # (29696,)
xvar = ind_cor_mat_broca; # (29696, 1398)

beta_y = np.linalg.lstsq(cvar, yvar, rcond=None)[0]     # shape (18,)
res_y = yvar - np.matmul(cvar, beta_y)      # shape: (29696,)
beta_x = np.linalg.lstsq(cvar, xvar, rcond=None)[0]     # shape (18, 1398)
res_x = xvar - np.matmul(cvar, beta_x)  # shape (29696, 1398)

# if res_x is just one dimension
if len(res_x.shape)>1: 
 np.corrcoef(res_x, res_y)[0,1]

# if it has more entries ...
else:
partcorr44 = np.zeros(res_x.shape[1]);              # (1398,)
for src_vertex_n in range(res_x.shape[1]):
  partcorr44[src_vertex_n] = np.corrcoef(res_x[:, src_vertex_n], res_y)[0,1]


## works the same as in matlab as in python
xvar = np.array([5,4,3,1,2])
cvar =  np.array([[1,2,3,4,5],[1,1,1,1,1]]) #[1,2,3,4,5; 1,1,1,1,1]
yvar  = np.array([1,4,1,5,1])

# matlab: partialcorr(x',y', c') -> -0.7655
# python pengouin_part_corr_copy(xvar, yvar, cvar.T) -> -0.7654530935859536
# python np.corrcoef(res_x, res_y)[0,1] -> -0.7654530935859536

xvar = np.array([[5,4,3,1,2], [4,3,3,1,2]])

# matlab: -0.7655; -0.9359
# python: array([-0.76545309, -0.93585673])

'''


'''
# xvar can have only one column/row (?), yvar can only have one, cvar can have any number again
# rows always reflect the vertex count
def pengouin_part_corr_copy(xvar, yvar, cvar):
    beta_y = np.linalg.lstsq(cvar, yvar, rcond=None)[0]     # shape (18,)
    res_y = yvar - np.matmul(cvar, beta_y)      # shape: (29696,)
    beta_x = np.linalg.lstsq(cvar, xvar, rcond=None)[0]     # shape (18, 1398)
    res_x = xvar - np.matmul(cvar, beta_x)  # shape (29696, 1398)
    if len(res_x.shape)==1: 
     partcorr44 = np.corrcoef(res_x, res_y)[0,1]
     # if it has more entries ...
    else:
     partcorr44 = np.zeros(res_x.shape[1]);              # (1398,)
     for src_vertex_n in range(res_x.shape[1]):
      partcorr44[src_vertex_n] = np.corrcoef(res_x[:, src_vertex_n], res_y)[0,1]
    return partcorr44
'''

# xvar can have any number of rows, yvar can only have one, cvar can have any number again
# does the partial correlation for each of the rows in xvar with the only availabel row in yvar, always controlling for all cvar rows
# its quite some messy code, but aparently it works ...
def pengouin_part_corr_copy(xvar, yvar, cvar):
    cvar_t = cvar.T
    beta_y = np.linalg.lstsq(cvar_t, np.atleast_2d(yvar).T, rcond=None)[0]     # shape (18,)
    res_y = yvar - np.matmul(cvar_t, beta_y).T      # shape: (29696,)
    beta_x = np.linalg.lstsq(cvar_t, np.atleast_2d(xvar).T, rcond=None)[0]     # shape (18, 1398)
    res_x = (xvar - np.matmul(cvar_t, beta_x).T).T   # shape (29696, 1398)
    if len(res_x.shape)==1: 
     partcorr44 = np.corrcoef(res_x, res_y)[0,1]
     # if it has more entries ...
    else:
     partcorr44 = np.zeros(res_x.shape[1]);              # (1398,)
     for src_vertex_n in range(res_x.shape[1]):
      partcorr44[src_vertex_n] = np.corrcoef(res_x[:,src_vertex_n], res_y.squeeze())[0,1]
    return partcorr44.squeeze()


