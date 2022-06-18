import matplotlib.pyplot as plt
import seaborn as sns

class plotter:
    def __init__(self, length, breath):
        self.plt = plt
        self.breath = breath
        self.length = length
        self.plt.figure(figsize=(length,breath))
        
    def barh(self, dataset1, dataset2, align, color):
        self.plt.barh(dataset1, dataset2, align=align, color=color)
        self.plt.gca().invert_yaxis()
        
    def xlabel(self, label):
        self.plt.xlabel(label)
    
    def title(self, title):
        self.plt.title(title)
        
    def savefig(self, name, dpi):
        self.plt.savefig(name, dpi=dpi)
        
    def set_seaborn(self, set=True):
        if set: sns.set()