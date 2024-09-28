# Bachelor_Thesis
This is the code for my Bsc thesis about Neural network based state estimation using Lie derivative.




### Introduction
This thesis focuses on problem of state estimation for **Non-linear Dynamics**. The problem with model-based observer is due to the need of solving PDEs. In this thesis i used **Gradient Decent (GD)** for updating lie derivatives and also by considering the observer dynamics as a Chen-Fliess series, the estimaton has minimum error. The proposed approach is demonstrated by oscillating system.

### Problem Formulation
For This Nonlinear system, Assume:

$$
\dot{x(t)} = F(x(t)),
y(t) = H(x(t))
$$

The observer suggest that by considering this linear form:

$$
\dot{z} = Az + By,
\hat{x} = T^{-1}(z)
$$

we can have an estimation about states. It is required that 1) (A, B) should be controllable 2) A should be Hurwitz and 3) T is can be inversed!
In additionally, it's required T to be obtain:

$$
\frac{\partial T}{\partial x}(x) F(x) = AT(x) + BH(x)
$$

Now the challenging part is due the part to solve above equation. The cost of solving Lie derivative of T along F(x) is high. So in this Thesis we define the observer we designed as **[Chen-Flies Series](https://github.com/iperezav/CFSpy)**:

$$
\dot{z} = g_0(z) + g_1(z)y + g_2(z)y + ... ,
\hat{x} = h(z)
$$

and

$$
Lie_g(h) = \frac{\partial h}{\partial z} g(z)
$$

and for estimation:

![image](https://github.com/user-attachments/assets/46030f61-b5dc-41e4-9c70-a6c96eb45b54)

also we used online Least-Squares for Chen-Fliess:

![image](https://github.com/user-attachments/assets/60f8f51c-c58f-4ce0-a616-e1d0d7ade6b1)

and

![image](https://github.com/user-attachments/assets/f600ceab-f7fc-43b3-8140-c09a9deec349)

the assumption shall be referred to as that of persistant excitation, Hence the optimal solution at time t + \Delta can be found:

![image](https://github.com/user-attachments/assets/83ab9554-721f-4de5-b1a9-f6258cc14bdf)

### result

![image](https://github.com/user-attachments/assets/e7196ac5-2d5e-4d16-9b08-349787b7e98b)



### Usage 

To use this code, you will need to install the following Python packages:

```
pip install numpy matplotlib
```

**This repository contains code for my Bsc thesis and for privacy, it's not the complete code, for more information contact nesaeian.mostafa@gmail.com**

1) run **Main.ipynb** for state estimation.

