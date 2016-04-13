import numpy as np

from independent.linear_chain import LinearChain

linearChain = LinearChain()
linearChain.fitFromFile()
evidMat = np.loadtxt('C:/Users/ckomurlu/Documents/workbench/experiments/20151121/temperature/outputs/LC-LINEAR/RND/' +
                     '0.2/evidences/evidMat_activeInf_model=lc-linear_T=12_trial=0_obsrate=0.2.csv', delimiter=',').\
    astype(np.bool_)

print evidMat[:, :3]

print linearChain.computeVar(evidMat=evidMat[:, :3])

pass
