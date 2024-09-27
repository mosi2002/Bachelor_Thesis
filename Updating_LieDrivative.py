import numpy as np

def index2multiindex(k, m):
    L = 0
    while (k >= (m + 1)**L):
        k -= (m + 1)**L
        L += 1
    li = []
    for i in range(L):
        li.append(k % (m + 1))
        k = k // (m + 1)  
    li.reverse()
    return li

def multiindex2index(li, m):
    k = sum([(m + 1)**i for i in range(len(li))])
    for i in range(len(li)):
        k += li[i] * (m + 1)**(len(li) - 1 - i)
    return k 

def trapez(u1, u2, dt = 1.0):
    # trapezoidal integral 
    s = 0.0
    s += u1[0] * u2[0] * 0.5 + u1[-1] * u2[-1] * 0.5
    for i in range(1, len(u1) - 1):
        s += u1[i] * u2[i]
    s *= dt
    return s

# def simpsons(u1, u2, dt=1.0):
#     n = len(u1)
    
#     s = u1[0] * u2[0] + u1[-1] * u2[-1]
#     for i in range(1, n-1, 2):
#         s += 4 * u1[i] * u2[i]
#     for i in range(2, n-1, 2):
#         s += 2 * u1[i] * u2[i]
#     return s * dt / 3.0


class Multiindex():
    def __init__(self, _list, _m, _N):
        self.list = _list
        self.m = _m
        self.index = multiindex2index(_list, _m)
        self.N = _N
        self.lie_derivative = 0.0
        self.recursive_integral = [0.0] * (_N + 1)
    def tail(self):
        return self.list[-1]
    def head(self):
        return self.list[0:-1]
    def length(self):
        return len(self.list)
    
    
    
class Series():
    def __init__(self, K, m, N):
        self.K = K
        self.m = m
        self.N = N 
        self.multiindex = []
        emptymultiindex = Multiindex([], m, N)
        self.multiindex.append(emptymultiindex)
        for length in range(K):
            for mu in self.multiindex:
                if len(mu.list) == length:
                    for i in range(m + 1):
                        newlist = mu.list + [i]
                        newmultiindex = Multiindex(newlist, m, N)
                        self.multiindex.append(newmultiindex)
        self.numberofindex = len(self.multiindex)
    
    def convolution(self, u, dt = 1.0):
        # convolution of input signals u with the recursive integrals for each multi-index.
        if (u.shape[0] == self.m):
            u = u.T
        u = np.hstack([np.ones((u.shape[0], 1)), u])
        for mu in self.multiindex:
            if (len(mu.list) == 0):
                mu.recursive_integral = [1.0] * u.shape[0]
            else:
                headmu = self.multiindex[multiindex2index(mu.head(), self.m)]
                for k in range(self.N):
                    mu.recursive_integral[k] = trapez(headmu.recursive_integral[0:(k+1)], u[0:(k+1), mu.tail()], dt)
    
    def regressor(self, x, y, dt = 1.0):
        self.convolution(y, dt)
        Phi = np.zeros((self.numberofindex, self.numberofindex))
        for i1 in range(self.numberofindex):
            for i2 in range(self.numberofindex):
                mu1 = self.multiindex[i1]
                mu2 = self.multiindex[i2]
                Phi[i1, i2] = trapez(mu1.recursive_integral, mu2.recursive_integral, dt)
        Phi = (Phi + Phi.T) * 0.5
        psi = np.zeros(self.numberofindex)
        for i in range(self.numberofindex):
            mu = self.multiindex[i]
            psi[i] = trapez(mu.recursive_integral, x, dt)
        c = trapez(x, x, dt) * 0.5
        return Phi, psi, c  
                
    def initialization_at_optimum(self, x, y, dt = 1.0):
        Phi, psi, c = self.regressor(x, y, dt)
        theta = np.linalg.solve(Phi, psi)
        for i in range(self.numberofindex):
            self.multiindex[i].lie_derivative = theta[i]
        obj = 0.5 * np.dot(theta, np.dot(Phi, theta)) - np.dot(theta, psi) + c
        return obj
    
    def initialization_at_unity(self, x, y, dt = 1.0):
        Phi, psi, c = self.regressor(x, y, dt)
        for i in range(self.numberofindex):
            if i == 0:
                self.multiindex[i].lie_derivative = 1.0
            else:
                self.multiindex[i].lie_derivative = 0.0
        theta = np.zeros(self.numberofindex)
        theta[0] = 1.0
        obj = 0.5 * np.dot(theta, np.dot(Phi, theta)) - np.dot(theta, psi) + c
        return obj
    
    def gradient_update(self, x, y, rate = 1.0, dt = 1.0):
        Phi, psi, c = self.regressor(x, y, dt)
        obj_old = 0.0 
        theta_old = np.array([mu.lie_derivative for mu in self.multiindex])
        obj_old = 0.5 * np.dot(theta_old, np.dot(Phi, theta_old)) - np.dot(theta_old, psi) + c
        vec = -rate * dt * (np.dot(Phi, theta_old) - psi)
        for i in range(self.numberofindex):
            self.multiindex[i].lie_derivative += vec[i]
        theta_new = np.array([mu.lie_derivative for mu in self.multiindex])
        obj_new = 0.5 * np.dot(theta_new, np.dot(Phi, theta_new)) - np.dot(theta_new, psi) + c
        return obj_old, obj_new
    
    def observe(self):
        xobs = 0.0
        for mu in self.multiindex:
            xobs += mu.lie_derivative * mu.recursive_integral[-1]
        return xobs