import os
import numpy as np
import math
from utils import OFv6

class datasetGen():
    def __init__(self, caseFolder,  lowbnd, highbnd, bounded = False, cwd = os.getcwd(), dirname = os.path.abspath(os.path.join(os.getcwd(), os.pardir))):
        # Path to the OpenFOAM case
        self.cwd = cwd
        self.dirname = dirname
        self.of_data = self.dirname + '/' + caseFolder
        self.ds_path = ''
        self.ds_name = caseFolder #'dataset_Re1e3'
        self.bounded = bounded
        self.lobound = lowbnd
        self.hgbound = highbnd
        
    def __call__(self):

        if not os.path.exists(self.ds_path):    
            folders = [fol.path.split('/')[-1] for fol in os.scandir(self.of_data+'/') if fol.is_dir()]
            if self.bounded:
                folders = sorted([int(f) for f in folders if f != '0' and f.isnumeric() and int(f)>=self.lobound and int(f)<=self.hgbound])
            else:
                folders = sorted([int(f) for f in folders if f != '0' and f.isnumeric()])
            of_reader = OFv6(self.of_data, name=self.ds_name + '_' + str(folders[0]) + '_' + str(folders[-1]), 
                             tStart=folders[0], tEnd=folders[-1], tStep=folders[1]-folders[0])
            ds = of_reader.dataset_generator()
            self.ds_path = self.cwd + '/' + self.ds_name + '_' + str(folders[0]) + '_' + str(folders[-1]) + '.npy'
            np.savetxt(self.ds_path, ds)
            print('Dataset is successfully saved in ' + self.ds_path + '.')
        else:
            ds = np.load(self.ds_path)
            print('Dataset already exists and loaded from ' +  self.ds_path + '.')

        return ds



