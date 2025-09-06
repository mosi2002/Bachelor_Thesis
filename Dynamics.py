import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from ChenFliess import *

def Osci(x, t, a=1.0, b=3.0):
    dx = np.zeros(2)
    dx[0] = a + x[0]**2 * x[1] - b*x[0] - x[0]
    dx[1] = b * x[0] - x[0]**2 * x[1]
    return dx

def OsciMeasurement(x):
    y = np.zeros(1)
    y[0] = x[0] + x[1]
    return y


def Lorentz(x, t, sigma=10.0, rho=28.0, beta=8.0/3.0, scale=10.0):
    dx = np.zeros(3)
    dx[0] = sigma*(x[1] - x[0])
    dx[1] = scale*x[0]*(rho/scale - x[2]) - x[1]
    dx[2] = scale*x[0]*x[1] - beta*x[2]
    return dx

def LorentzMeasurement(x):
    y = np.zeros(1)
    y[0] = x[1]
    return y

import numpy as np

def inverted_pendulum_dynamics(x, t, M=1.0, m_p=0.1, l=1.0, g=9.81):
    # Unpack state variables
    x1, x2, theta, theta_dot = x
    
    # Compute sin and cos of theta for convenience
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Compute the equations of motion
    denominator = M + m_p * sin_theta**2
    
    x1_ddot = ( m_p * sin_theta * (l * theta_dot**2 - g * cos_theta)) / denominator
    theta_ddot = (g * sin_theta - cos_theta * (x1_ddot - l * theta_dot**2 * sin_theta)) / (l * (4/3 - m_p * cos_theta**2 / (M + m_p)))
    
    # Pack the derivatives into an array
    dx = np.zeros(4)
    dx[0] = x2
    dx[1] = x1_ddot
    dx[2] = theta_dot
    dx[3] = theta_ddot
    
    return dx

def inverted_pendulum_measurement(x):
    # For example, we can measure the position of the cart and the angle of the pendulum
    y = np.zeros(2)
    y[0] = x[0]  # Cart position
    y[1] = x[2]
    return y


class Plant:
    def __init__(self, nx, ny, dynamics=Osci, measurement=OsciMeasurement):
        self.nx = nx
        self.ny = ny
        self.dynamics = dynamics
        self.measurement = measurement
        self.x = np.zeros(nx)
        self.y = self.measurement(self.x)
    def initialize(self, x0):
        self.x = x0
        self.y = self.measurement(self.x)
    def move_forward(self, t=0.0, dt=1.0):
        r=0.05
        self.x += dt * self.dynamics(self.x, t)
        self.y = self.measurement(self.x)
    
    
class Observer:
    def __init__(self, nx, ny, maxlength=3, horizon=10):
        self.nx = nx
        self.ny = ny
        self.maxlength = maxlength
        self.horizon = horizon
        self.x = np.zeros(nx)
        self.xhistory = np.zeros((self.horizon + 1, self.nx))
        self.y = np.zeros(ny)
        self.yhistory = np.zeros((self.horizon + 1, self.ny))
        self.series = []
        
    def create(self):
        for i in range(self.nx):
            newseries = Series(self.maxlength, self.ny, self.horizon)
            self.series.append(newseries)
                
    def measure(self, plant):
        self.x = plant.x
        self.xhistory = np.delete(self.xhistory, 0, axis=0)
        self.xhistory = np.vstack([self.xhistory, self.x])
        self.y = plant.y
        self.yhistory = np.delete(self.yhistory, 0, axis=0)
        self.yhistory = np.vstack([self.yhistory, self.y])
    
        
    def initialize(self, dt=1.0):
        for i in range(self.nx):
            obj = self.series[i].initialization_at_unity(self.xhistory[:,i], self.yhistory, dt)
    
    def update(self, rates, dt = 1.0):
        obj_old_vec = np.zeros(self.nx)
        obj_new_vec = np.zeros(self.nx)
        xobs_vec = np.zeros(self.nx)
        for i in range(self.nx):
            obj_old, obj_new = self.series[i].gradient_update(self.xhistory[:,i], self.yhistory, rates[i], dt)
            obj_old_vec[i] = obj_old
            obj_new_vec[i] = obj_new
            xobs = self.series[i].observe()
            xobs_vec[i] = xobs
        return obj_old_vec, obj_new_vec, xobs_vec
    
    
class Simulator:
    def __init__(self, model='Osci', observerrateconstant=1.0):
        # Model information
        if model == 'Osci':
            self.nx = 2
            self.ny = 1
            self.dynamics = Osci
            self.measurement = OsciMeasurement
            self.dt = 0.02
            self.Nt = 2500

        if model == "inverted":
            self.nx = 4
            self.ny = 2
            self.dynamics = inverted_pendulum_dynamics
            self.measurement = inverted_pendulum_measurement
            self.dt = 0.01
            self.Nt = 1000

        if model == 'Lorentz':
            self.nx = 3
            self.ny = 1
            self.dynamics = Lorentz
            self.measurement = LorentzMeasurement
            self.dt = 0.01
            self.Nt = 5000
        # Observer hyperparameters
        self.observermaxlength = 6
        self.observerhorizon = 10
        self.observerwaittime = 0
        self.observerrates = self.nx * [observerrateconstant / self.dt]
        self.plant = Plant(nx=self.nx, ny=self.ny, dynamics=self.dynamics, measurement=self.measurement)
        self.observer = Observer(nx=self.nx, ny=self.ny, maxlength=self.observermaxlength, horizon=self.observerhorizon)
        self.observer.create()
        # Records
        self.Nt += self.observerhorizon + self.observerwaittime - 1
        self.xtrajectory = np.zeros((self.Nt + 1, self.nx))
        self.ytrajectory = np.zeros((self.Nt + 1, self.ny))
        self.principalstrajectory = np.zeros((self.Nt + 1, self.nx))
        self.principalsposteriortrajectory = np.zeros((self.Nt + 1, self.nx))
        self.xobstrajectory = np.zeros((self.Nt + 1, self.nx))
        self.Joldtrajectory = np.zeros((self.Nt + 1, self.nx))
        self.Jnewtrajectory = np.zeros((self.Nt + 1, self.nx))
        
    def simulate(self, x0 = np.array([1.0, 1.0]), xinfo=True):
        x0 = x0
        if (len(x0) != self.nx):
            x0 = np.array(self.nx * [x0[0]])
        self.plant.initialize(x0)
        self.xtrajectory[0, :] = self.plant.x
        self.ytrajectory[0, :] = self.plant.y
        self.observer.initialize(dt=self.dt)
        J = 0.
        if xinfo:
            self.observerwaittime = 0
        else:
            self.observerwaittime = 500

        for tcount in range(1, self.Nt + 1):
            # Time moves forward
            self.plant.move_forward(t=tcount*self.dt, dt=self.dt)
            self.xtrajectory[tcount, :] = self.plant.x
            self.ytrajectory[tcount, :] = self.plant.y
        
            if tcount >= (self.observerhorizon + self.observerwaittime):
                # Reduce dimensionality
                
                self.observer.measure(self.plant)
                # Observer update
                obj_old_vec, obj_new_vec, xobs_vec = self.observer.update(rates=self.observerrates, dt=self.dt)
                self.Joldtrajectory[tcount, :] = obj_old_vec
                self.Jnewtrajectory[tcount, :] = obj_new_vec
                self.xobstrajectory[tcount, :] = xobs_vec
                print(f"tcount: {tcount}, Jnewtrajectory: {self.Jnewtrajectory[tcount, :]}")
            if (tcount % 100 == 0):
                print('Sampling time', tcount)
        
        # CALCULATE THE PERFORMANCE
        if xinfo:
            J_trajectory = []
            for tcount in range(self.observerhorizon + self.observerwaittime, self.Nt + 1):
                J += np.linalg.norm(self.xobstrajectory[tcount, :] - self.xtrajectory[tcount, :], 2)**2
                J_value = np.sqrt(J/(self.Nt + 1 - self.observerhorizon - self.observerwaittime))
                J_trajectory.append(J_value)
            print(len(J_trajectory))
            J = J_trajectory[-1]    
                # print(J) 
        else:
            for tcount in range(self.observerhorizon + self.observerwaittime, self.Nt + 1):
                self.principalsposteriortrajectory[tcount, :] = p
                J += np.linalg.norm(self.xobstrajectory[tcount, :] - p, 2)**2 
        # J = np.sqrt(J/(self.Nt + 1 - self.observerhorizon - self.observerwaittime))
        return J
        
        
    def plot(self, xinfo=True):
        trange = np.arange(self.Nt + 1 - self.observerhorizon - self.observerwaittime) * self.dt
        
        # Plot the x components separately
        fig, ax = plt.subplots()
        for i in range(self.nx):
            ax = plt.subplot(self.nx, 1, i + 1)
            if xinfo:
                ax.plot(trange, self.xtrajectory[self.observerhorizon + self.observerwaittime:, i], 'b', label=r'$x_{%s}$' % (i+1))
                ax.plot(trange, self.xobstrajectory[self.observerhorizon + self.observerwaittime:, i], 'r', label=r'$\hat{x}_{%s}$' % (i+1))
            else:
                ax.plot(trange, self.principalsposteriortrajectory[self.observerhorizon + self.observerwaittime:, i], 'k', label=r'$\pi_{%s}^\ast$' % (i+1))
                ax.plot(trange, self.principalstrajectory[self.observerhorizon + self.observerwaittime:, i], 'b', label=r'$\pi_{%s}$' % (i+1))
                ax.plot(trange, self.xobstrajectory[self.observerhorizon + self.observerwaittime:, i], 'r', label=r'$\hat{\pi}_{%s}$' % (i+1))
            ax.legend()
            if i == self.nx - 1:
                ax.set_xlabel(r'$t$')
        figurename = 'one'
        fig.savefig(figurename, format='eps', dpi=1200)
        plt.show()
        # Save x components to csv file
        if xinfo:
            csvdata = np.hstack([trange.reshape((-1,1)), self.xtrajectory[self.observerhorizon + self.observerwaittime:, :], self.xobstrajectory[self.observerhorizon + self.observerwaittime:, :]])
            header = 't' + ',x'*self.nx + ',xhat'*self.nx
        else:
            csvdata = np.hstack([trange.reshape((-1,1)), self.principalsposteriortrajectory[self.observerhorizon + self.observerwaittime:, :], self.principalstrajectory[self.observerhorizon + self.observerwaittime:, :], self.xobstrajectory[self.observerhorizon + self.observerwaittime:, :]])
            header = 't' + ',pistar'*self.nx + ',pi'*self.nx + ',pihat'*self.nx
        csvname = 'two'
        np.savetxt(csvname, csvdata, delimiter=",", header=header)

        
    
    
   