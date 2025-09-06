import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from scipy.integrate import odeint
from Dynamics import * 
    
# Standard tuning for Oscillator: model='Osci', observerrateconstant=1.0, kpcanu=0.05, kpcabeta=1.0, kpcahp=0.5
# Standard tuning for Lorentz: model='Lorentz', observerrateconstant=5.0, kpcanu=0.05, kpcabeta=1.0, kpcahp=0.5
xinfo = True
sim = Simulator(model='Osci', observerrateconstant=1.0)
J = sim.simulate(x0 = np.array([1.0, 1.0]), xinfo=xinfo)
print('Averaged observation error =', J)
sim.plot(xinfo=xinfo)
