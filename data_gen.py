import numpy as np

class CosineData:
    def __init__(self, ll, ul, mu, sig, noise_sig = 0.05, num=25):
        #Sanity checks
        if ll>ul:
            print("The lower limit is larger than the upper limit.")
        if len(mu) != len(sig):
            print("Dimension mismatch in mu and sigma.")
        
        self.ll = ll
        self.ul = ul
        self.mu = mu
        self.sig = sig
        self.num = num
        self.noise_sig = noise_sig

    def generate(self):
        self.X = []
        self.y = []
        n = 0
        while n<self.num:
            i = np.random.randint(0, len(self.mu))
            x_r = np.random.normal(self.mu[i], self.sig[i])
            self.X.append(x_r)
            self.y.append(np.cos(x_r) + np.random.normal(0, self.noise_sig, 1)[0])
            n = n+1
        self.X, self.y = (list(t) for t in zip(*sorted(zip(self.X, self.y))))

    def get_rn(self):
        return self.X, self.y
