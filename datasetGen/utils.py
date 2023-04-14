import numpy as np

class OFv6:
    def __init__(self, path, tStart, tEnd, tStep):
        self.path = path
        self.tStart = tStart
        self.tEnd = tEnd
        self.tStep = tStep
        self.dataset = []

    def reader(self, file):
        
        with open(file) as f:
            ls = f.readlines()
        f.close()

        skip = 20
        numSample = int(ls[skip])
        if file.split('/')[-1] == 'S_ij' or file.split('/')[-1] == 'U':
            data = [[v for v in np.float_(ls[l][1:-2].split())] for l in range(skip+2, numSample+skip+2)]
        elif file.split('/')[-1] == 'Cs':
            data = [[v for v in np.float_(ls[l].split())] for l in range(skip+2, numSample+skip+2)]

        return data

    def data_collector(self, path):
        
        Sij = self.reader(path + '/S_ij')
        U = self.reader(path + '/U')
        Cs = self.reader(path + '/Cs')

        return np.concatenate((U,Sij,Cs),axis=1)

    def dataset_generator(self):

        t = self.tStart
        while t <= self.tEnd:
            path = self.path + '/' + str(t)
            if t == self.tStart:
                self.dataset = self.data_collector(path)
            else:
                self.dataset = np.concatenate((self.dataset,self.data_collector(path)),axis=0) 
            print('Time: ' + str(t) + ' is done')
            t = t + self.tStep        

        print('Dataset shape: ' + str(self.dataset.shape))

        return self.dataset

