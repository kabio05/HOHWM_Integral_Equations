## Introduction

### The Definition of Integral Equations

The term integral equation was first used by Paul du Bois-Reymond in 1888 for equations in which an unknown function occurs under an integral.[^1] 

A standard integral equation has the form:[^2]
$$
u(x) = f(x) + \int^{\beta(x)}_{\alpha(x)} K(x,t)\Phi(u(t))dt,
$$

where $u(x)$ is an unknown function called the solution of the integral equation, $K(x,t)$ is called the kernel of the integral equation, and $K, \Phi, \alpha, \beta, f$ are given functions.

In this project, we are mainly interested in two types of integral equations, [^2]

- Fredholm Integral Equations

$$
u(x) = f(x) + \lambda\int^{b}_{a} K(x,t)\Phi(u(t))dt,\  a \leq x, \  t \leq b,
$$

where the limits of integration $a$ and $b$ are constants.

- Volterra Integral Equations

$$
u(x) = f(x) + \lambda\int^{x}_{0} K(x,t)\Phi(u(t))dt,
$$

where the limits of integration are functions of x rather than constants.

Linearity: An integral equation is linear if $\Phi$ is linear, i.e. the power of $u(x)$ is one, and nonlinear otherwise.



### The  Applications of Integral Equations

Many Physic and engineering applications are usually described by integral equations. A variety of initial and boundary value problems that can be transformed into Fredholm or Volterra integral equations. For example, the initial value problem
$$
u'(x) = 2xu(x), \ x\geq0.
$$
with the initial condition $u(0) = 1$, can be easily converted by integrating both sides with respect to $x$ from 0 to x:
$$
\int^{x}_{0} u'(t)dt = \int^{x}_{0} 2tu(t)dt,
$$
substituting the initial condition $u(0) = 1$ into the left hand side we get
$$
u(x) =1 + \int^{x}_{0} 2tu(t) dt,
$$
which is a linear Volterra equations.

The example illustrates the close connection between partial differential equations and integral equations. This connection signifies that integral equations find widespread application in various fields, including radiative transfer, electrostatics, electromagnetic scattering, and the propagation of acoustical and elastic waves. Additionally, integral equations served as the foundational framework in the early stages of the development of functional analysis, providing an abstract foundation for the study of integral (and differential) equations.

In contemporary mathematical research, particularly in the context of inverse and ill-posed problems, integral equations continue to play a pivotal role. They are indispensable in domains such as medical imaging and nondestructive evaluation. Integral equations also emerge as essential tools in solving the problems of diffraction phenomena, conformal mapping, water wave analysis, quantum mechanical scattering, and Volterra's population growth model. These diverse applications show the significance of the study of integral equations.

### Why we need the numerical methods to solve the integral equations

For most of concrete integral equations, analytical solutions are hard to find. Due to this fact, several numerical methods have been
developed for finding numerical solutions of integral equations. Examples include Chebyshev polynomials, Taylor series method, the wavelet method, Bernstein's approximation method, the modified homotopy perturbation method, the Toeplitz matrix method, and other methods. In this project, we will concentrate on the wavelets methods.

The wavelet is a mathematical function used to divide a target function over an interval into different scale components. There are a wide range of various wavelet transforms suitable for different application. In particular, Haar wavelet (HWM) is a common and well-known wavelet basis. Haar wavelets are the simplest orthonormal wavelet with compact support and they have been used in different numerical approximation problems.[^6] The problem of Haar wavelet is that it is slow to converge for numerical approximation. To improve the convergence rate of Haar wavelet, a method called Higher order Haar wavelet method (HOHWM) was introduced. [^3] HOHWN transform the integral functions into a system of linear/nonlinear algebraic equations. In this project, we will focus on applying HOHWM on the Fredholm and Volterra Integral Equations and develop an efficient iterative solver to solve the arising linear system. 

## Main

### The Definition of Haar wavelet functions

The Haar wavelet family defined on the interval $[a, b)$ consists of the following functions: [^4]
$$
h_{1}(x) = \begin{cases}
1 & \text{for} & x \in [a,b)\\
0 & \text{elsewhere.}
\end{cases}
$$
And
$$
h_{i}(x) = \begin{cases}
1  & \text{for} \quad x \in [\xi_{1},\xi_{2})\\
-1 & \text{for} \quad x \in [\xi_{2},\xi_{3})\\
0  & \text{elsewhere.} & i = 2,3,...,
\end{cases}
$$
where
$$
\xi_{1} = a + (b-a)\frac{k}{m}, \quad \xi_2 = a + (b-a)\frac{k+0.5}{m}, \quad \xi_{3} = a + (b-a)\frac{(k+1)}{m}; \\
m = 2^{j}, \quad j = 0,1, \ldots, J, \ J=2^{M},\quad k= 0, 1, \ldots, m-1, \quad i = 2,3, \ldots,2M.
$$
The integer j indicates the level of the wavelet and k is the translation parameter. The maximal level of resolution is $J$ . The relation between $i$, $m$ and $k$ is given by $i = m + k + 1$. The function $h_1(x)$ is called scaling function whereas $h_2(x)$ is the mother wavelet for the Haar wavelet family. The parameter $m = 2j $ $ (M = 2J)$ corresponds to a maximum number of square waves can be sequentially deployed in interval $[a,b)$ and the parameter $k$ indicates the location of the particular square wave.[^3]

<img src="C:\Users\Yi Zong\Desktop\M4R_Files\M4R_CODE\Haars.png" style="zoom:25%;" />

We can see the superposition of the Haar wavelet functions to get a intuitive understanding of such kind of functions.

<img src="C:\Users\Yi Zong\Desktop\M4R_Files\M4R_CODE\Haars_Superposition.png" style="zoom: 25%;" />

### Some Properties of Haar wavelet functions

We can directly get the fact from the figure that the integrals of the Haar wavelet functions zeros:
$$
\int^{a}_{b} h_{i}(x) dx = 0, \quad i = 2,3, \ldots
$$
 Also, the Haar wavelet functions are orthogonal to each other, in analytic description, it is:
$$
\int^{b}_{a} h_{j}(x)h_{k}(x)dx = 
\begin{cases}
(b-a)2^{-j} & \text{when} \quad j = k \\
0 & \text{when} \quad j \neq k.
\end{cases}
$$


Thus any function $f(x)$ which is square integrable in the interval $(a,b)$ can be expressed as an infinite sum of Haar wavelets [^6]
$$
f(x) = \sum^{\infty}_{i = 1} a_{i}h_{i}(x).
$$
An essential shortcoming of the Haar wavelets is that they are not continuous. The derivatives do not exist in the points of discontinuity, therefore it is not possible to apply the Haar wavelets directly to solving differential equations. [^4]

A method to solve this problem was proposed by Chen and Hsiao [^7][^8]. They recommended to expand into the Haar series not the function itself, but its highest derivative appearing in the differential equation; the other derivatives (and the function) are obtained through integrations. All these ingredients are then incorporated into the whole system, discretized by the Galerkin or collocation method.

Chen and Hsiao demonstrated the possibilities of their method by solving linear systems of ordinary differential equations (ODEs) and partial differential equations (PDEs). This method was introduced by Lepik as a method called Higher order Haar wavelet method (HOHWM)[^4]. We will try to apply this method into the integral functions.

### The Higher order Haar wavelet method (HOHWM) 

For the HOHWM, we consider the $sth$ derivative of the unknown function $u(x)$ as an approximation of the sum of Haar wavelets[^5]
$$
\frac{d^{s}u(x)}{dx^{s}} = \sum^{N}_{i=1} a_{i}h_{i}(x), \quad s = 1,2,\ldots,
$$
where $a_{i}$, $i = 1,2,\ldots,N$ are N unknown constants. The function $u(x)$ is obtains by integrating the summation $s$ times, which will lead to additional s constants. To calculate the Haar coefficients we consider the collocation points 
$$
x_{k} =  a + (b-a)\frac{k-0.5}{N}, \quad k = 1,2,\ldots,N.
$$
By substitution for the Haar wavelets with the integral equation and using the above collocation points for discretization, we will get an $N \times (N+s)$ linear or nonlinear system, which can be solved by the linear solver. 

### Numerical implementation

$$
S_{1}(j) = \frac{1}{N}\sum^{N}_{k=1}K(0,t_{k})p_{j,1}(t_{k}), \quad 
S_2 = \frac{1}{N}\sum^{N}_{k=1}K(0,t_{k})
$$

$$
\sum^{N}_{j=1}a_{j}\left(p_{j,1}(x_{i}) - \frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})p_{j,1}(t_{k})\right)\\
= f(x_{i}) - \frac{1}{1-S_{2}}
\left(1 - \frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})\right)
\left(f(0) + \sum^{N}_{j=1}a_{j}S_{1}(j)\right), \quad i = 1,2,...N.
$$

For convenience, we introduce the notation:
$$
\begin{align*}
& A := p_{j,1}(x_{i}) - \frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})p_{j,1}(t_{k})\\ 
& B := 1 - \frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})
\end{align*}
$$
Note that A is an $N \times N$ matrix and B is an N-vector. Then we have
$$
\sum^{N}_{j = 1}a_{j} A + \frac{1}{1-S_{2}} B \sum^{N}_{j=1}a_{j}S_{1}{(j)}\\
= f(x_{i}) - \frac{f(0)}{1-S_{2}}B
$$
And we find RHS is an N-vector. Since the summation can be exchanged, we have
$$
\text{LHS} = \sum^{N}_{j = 1} a_{j}\left[ A + \frac{S_{1}(j)}{1-S_{2}}B  \right ]
$$
And we find LHS is in the form $\text{Matrix} \times \text{vector}$, then the problem become an least square problem.

Call
$$
C := A + \frac{S_{1}(j)}{1-S_{2}}B
$$


Expand $C$, we have
$$
C = 
p_{j,1}(x_{i}) - \frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})p_{j,1}(t_{k}) + \frac{S_{1}(j)}{1-S_{2}}\left(1 - \frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})\right)
$$

$$
C = p_{j,1}(x_{i})
+\mathbb{1}\frac{S_{1}(j)}{1-S_{2}}
- \left(
 \frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})p_{j,1}(t_{k})+
\frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})\frac{S_{1}(j)}{1-S_{2}}
\right)
$$

$$
C = p_{j,1}(x_{i})
+\mathbb{1}\frac{S_{1}(j)}{1-S_{2}}
- \left[
\frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})\left(p_{j,1}(t_{k}) + \frac{S_{1}(j)}{1-S_{2}} \right)
\right]
$$

where $\mathbb{1}$ is an $N \times 1$ vector. We show that $C$ is exactly a matrix.

Define
$$
D := p_{j,1}(x_{i}) \\
E := \mathbb{1}\frac{S_{1}(j)}{1-S_{2}}\\
F :=  \left[
\frac{1}{N}\sum^{N}_{k=1}K(x_{i},t_{k})\left(\frac{S_{1}(j)}{1-S_{2}} - p_{j,1}(t_{k})\right)
\right] \\
C = D + E + F
$$


## Reference

[^1]: M. R. Dennis, P. Glendinning, P. A. Martin, F. Santosa, and J. Tanner, eds., The Princeton companion to applied mathematics, Prince-ton University Press, Princeton, NJ, 2015, pp. 200.]
[^2]: Wazwaz, Abdul-Majid (2005). *A First Course in Integral Equation*. World Scientific.
[^3]: J. Majaka, M. Pohlaka, K. Karjusta, M. Eermea, J. Kurnitskia, B.S. Shvartsman, New higher order Haar wavelet method: Application to FGM structures, Composite Structures, 201 (2018), pp. 72-78.
[^4]: U. Lepik, Application of the haar wavelet transform to solving integral and differential equations., in Proceedings of the Estonian Academy of Sciences, Physics, Mathematics, vol. 56, 2007.
[^5]: S. Yasmeen and R. Amin, Higher order haar wavelet method for numerical solution of integral equations, Computational and Applied Mathematics, 42 (2023), p. 147.
[^6]: I. Aziz, F. Haq, et al., A comparative study of numerical integration based on haar wavelets and hybrid functions, Computers & Mathematics with Applications, 59 (2010), pp. 2026–2036.
[^7]: Chen, C. F. and Hsiao, C.-H. Haar wavelet method for solving lumped and distributed-parameter systems. IEE Proc. Control Theory Appl., 1997, 144, 87–94.
[^8]: Chen, C. F. and Hsiao, C.-H. Wavelet approach to optimising dynamic systems. IEE Proc. Control Theory Appl., 1997, 146, 213–219.
