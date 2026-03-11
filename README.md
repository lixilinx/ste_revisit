# The straight-through estimator (STE): a revisit
It's common to encounter discontinuous function $y=f(x)$ like quantization, step and sign functions, clustering, etc. There exists many choices to pass through the gradients to offer different tradeoffs between bias and variance, say continuous relaxation like Gumbel-Softmax and estimators like REINFORCE and straight-through. But, if the derivative of $f(\cdot)$ can be expressed with the Dirac delta function, generally we could get improved gradient estimator by passing through the gradient as

$$ E_{v\sim p_V(v)}\left[ \frac{\partial f(x-v)}{\partial x} \right\]  = \frac{\partial f(x)}{\partial x} \circledast p_V(x) $$

where $v$ is drawn from a distribution with pdf $p_V(\cdot)$, $E_v[\cdot]$ is replaced with sample average during training and $\circledast$ denotes convolution. This technique is widely applicable, for both scalar- and vector-valued functions. Here, we focus on its application to the straight-through estimator (STE).       

## Example 1: recovery of the STE
Let the target function be $y=f(x)={\rm round}(x)$. Its derivative is $\frac{dy}{dx} = \ldots +\delta(x+1.5) + \delta(x+0.5) + \delta(x-0.5) +\delta(x-1.5)+ \ldots$. Let $v\sim \mathcal{U}(-0.5, 0.5)$, i.e., $p_V(v)=I(-0.5\le v< 0.5)$. Then, we have, 

$$ 
\begin{aligned}
E_{v\sim \mathcal{U}(-0.5, 0.5)}\left[ \frac{\partial f(x-v)}{\partial x} \right\] & = \ldots + \int_v \delta(x+0.5-v)I(-0.5\le v< 0.5)dv + \int_v \delta(x-0.5-v)I(-0.5\le v< 0.5)dv + \ldots \\
& = \ldots + I(-0.5\le x+0.5<0.5) + I(-0.5\le x - 0.5 < 0.5) + \ldots \\
& = \ldots + I(-1\le x<0) + I(0\le x < 1) + \ldots = 1
\end{aligned}$$

This is the STE. The stochastic rounding noise reduces the bias of STE. We can treat the Dirac delta function as a legitimate function and pass through any order of derivatives in this way. Clearly, with $p_V(v)=\delta(v)$, we recover the original derivative. 

## Example 2: STE for the sign function

Derivative of the sign function $y={\rm sign}(x)$ is $\frac{dy}{dx} = 2\delta(x)$. We let $v\sim U(-a, a)$, i.e., $p_V(v) = \frac{1}{2a}I(-a\le v<a)$, where $a>0$. Then, it's ready to show that $E_{v\sim U(-a, a)}\left[ 2 \delta(x-v) p_V(v)\right] = I(-a\le x<a)/a \propto I(-a\le x<a)$. This example suggests that the STE applies to the sign function only for $x$ in a compact set like $x\in [-a, a]$. Intuitively, it's not possible to effectively regularize the training with finite amount of stochastic rounding noise if $x$ can take arbitrarily large values.  

## Example 3: STE for the Nvidia FP4 quantization function 

The Nvidia FP4 format has values $\\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\\}$. Different from example 1, these values are not equally spaced. Hence, it's not obvious to design a $p_V(\cdot)$ such that the smoothed derivative becomes a constant. Still, we can design a monotonic mapping $g(\cdot)$ from set 

$$\\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\\}$$ 

to set 

$$\\{0, \pm 1, \pm 2, \pm 3, \pm 4, \pm 5, \pm 6, \pm 7\\}$$

and reuse the math in example 1. Basically, we redefine $f(\cdot)$ as $y=g^{-1}(r(g(x)))=g^{-1}\circ r \circ g(x)$, where $r$ is the rounding function and $g^{-1}$ is the inverse function of $g$. Then, we get derivative $\frac{dy}{dx}=g^{-1\'}\circ r^{'} \circ g^{'}(x)$. the derivatives of $g$ and its inverse roughly cancel out each other and the derivative of $r$ smoothes out to a constant as in example 1. In this way, the STE still applies.        

Run [this script](https://github.com/lixilinx/ste_revisit/blob/main/fp4_example.py) to generate the following image to illustrate this idea. Piece-wise linear functions are used for the mappings between the two sets in this example (any proper monotonic mapping can be good). Again, the STE here can only be valid for $x$ in a compact set.   
![fp4](https://github.com/lixilinx/ste_revisit/blob/main/fp4_example.svg)

## Example 4: application to encoder latent vector compression 

[This script](https://github.com/lixilinx/ste_revisit/blob/main/binary_code_example.py) gives an example on compressing each dim of the latent vector of a simple codec trained for the CIFAR10 images to 1-bit. A straightforward Pytorch implementation of the sign function for STE is 

$$ y = x - (x - {\rm torch.sign}(x)).{\rm detach}() $$

However, there is no stochastic noise to regularize the training and $x$ can be arbitrarily large. Following the discussion of example 2, we can first constrain $x$ to a compact set and then take its sign as below  

$$
\begin{aligned}
x & \leftarrow x/\sqrt{1+x^2} \\
y & = x - (x - {\rm torch.sign}(x - {\rm noise\\_level}*({\rm torch.rand\\_like}(x) - 0.5))).{\rm detach}()
\end{aligned}
$$

where $0\le {\rm noise\\_level}\le 2$ and we should anneal it to a small enough number when approaching convergence. We have tried two optimizers and both benefit from the improved STE. 

![binary_code](https://github.com/lixilinx/ste_revisit/blob/main/binary_code_example.svg)




