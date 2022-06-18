
import sys
from telnetlib import SE
import h5py

class Saver:
    def __init__(self, list, stdout):
        self.list = list
        self.stdout = stdout
        
    def saveCSV(self, filename):
        with open(filename, 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(self.list)
            sys.stdout = self.stdout # Reset the standard output to its original value
    
    def saveHDF5(self, filename):
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("default", dtype="S15", data = self.list)
        