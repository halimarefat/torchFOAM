import os
import numpy as np
from datasetGen import datasetGen

np.random.seed(10)

ds = datasetGen(caseFolder='_dynSmagCase_Re3e3', lowbnd=600, highbnd=1200, bounded=True) 
ds.__call__()

