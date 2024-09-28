# Bachelor_Thesis
This is the code for my Bsc thesis about Neural network based state estimation using Lie derivative.




### Introduction
This thesis focuses on problem of state estimation for **Non-linear Dynamics**. The problem with model-based observer is due to the need of solving PDEs. In this thesis i used **Gradient Decent (GD)** for updating lie derivatives and also by considering the observer dynamics as a Chen-Fliess series, the estimaton has minimum error. The proposed approach is demonstrated by oscillating system.

### Problem Formulation
For This Nonlinear system, Assume:

$$
\dot{x(t)} = F(x(t)) \\
y(t) = H(x(t))
$$



### Usage 

To use this code, you will need to install the following Python packages:

```
pip install numpy matplotlib
```

1) run **Main.ipynb** for state estimation.

