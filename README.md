# A revisit of the straight-through estimator (STE)
It's common to encounter discontinuous function $y=f(x)$ like quantization, step and sign functions, clustering, etc. There exists many choices to pass through the gradients to offer different tradeoffs between bias and variance, say continuous relaxation like Gumbel-Softmax and estimators like REINFORCE and straight-through. But, if the derivative of $f(\cdot)$ can be expressed with the Dirac delta function, generally we could get less biased and lower variance gradient estimator by passing through the gradient as

$$ E_{v\sim p_V(v)}\left[ \frac{\partial f(x-v)}{\partial x} \right\] $$

where $v$ is drawn from a pdf $p_V(\cdot)$ and $E_v[\cdot]$ is replaced with sample average during training. This technique is widely applicable, for both scalar- and vector-valued functions. Here, we focus on its application to the straight-through estimator (STE).       

## Example 1: recovery of the STE
Let the target function be $y=f(x)={\rm round}(x)$. Its derivative is $\frac{dy}{dx} = \ldots +\delta(x+1.5) + \delta(x+0.5) + \delta(x-0.5) +\delta(x-1.5)+ \ldots$. Let $v\sim \mathcal{U}(-0.5, 0.5)$, i.e., $p_V(v)=I(-0.5\le v< 0.5)$. Then, we have, 

$$ 
\begin{aligned}
E_{v\sim \mathcal{U}(-0.5, 0.5)}\left[ \frac{\partial f(x-v)}{\partial x} \right\] & = \ldots + \int_v \delta(x+0.5-v)I(-0.5\le v< 0.5)dv + \int_v \delta(x-0.5-v)I(-0.5\le v< 0.5)dv + \ldots \\
& = \ldots + I(-0.5\le x+0.5<0.5) + I(-0.5\le x - 0.5 < 0.5) + \ldots \\
& = \ldots + I(-1\le x<0) + I(0\le x < 1) + \ldots = 1
\end{aligned}$$

This is the STE. The injected noise reduces the bias of STE. We can treat the Dirac delta function as a legitimate function and pass through any order of derivatives in this way. Clearly, with $p_V(v)=\delta(v)$, we recover the original derivative. 

## Example 2: STE for the sign function

Derivative of the sign function $y={\rm sign}(x)$ is $\frac{dy}{dx} = 2\delta(x)$. We let $v\sim U(-a, a)$, i.e., $p_V(v) = \frac{1}{2a}I(-a\le v<a)$, where $a>0$. Then, it's ready to show that $E_{v\sim U(-a, a)}\left[ 2 \delta(x-v) p_V(v)\right] = I(-a\le x<a)/a \propto I(-a\le x<a)$. This example suggests that the STE applies to the sign function only for $x$ in a compact set like $x\in [-a, a]$. Intuitively, any finite rounding noise could be ineffective to regularize the training if $x$ can take arbitrarily large values. Hence, the STE can only be valid for $x$ in a compact set. 

## Example 3: STE for the Nvidia FP4 quantization function 

The Nvidia FP4 can take values $\\{-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6\\}$. Different from example 1, these values are not equally spaced. Hence, it's not obvious to design a $p_V(\cdot)$ such that the smoothed derivative becomes a constant. Still, we can design a monotonic mapping from set 

$$\\{-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6\\}$$ 

to set 

$$\\{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7\\}$$

and reuse the math in example 1 to make the function $y=f(x)$ compatible with the STE.   

Run [this script](https://github.com/lixilinx/ste_revisit/blob/main/fp4_example.py) to generate the following image to illustrate this idea. Piece-wise linear functions are used for the mappings between the two sets in this example (surely, any proper monotonic mapping can be good). Again, the STE can only be valid for $x$ in a compact set.   
![fp4](https://github.com/lixilinx/ste_revisit/blob/main/fp4_example.svg)

## Example 4: application to encoder latent variable tokenization 

[This script](https://github.com/lixilinx/ste_revisit/blob/main/binary_code_example.py) gives an example on how to tokenize the latent variable of a simple codec trained for the CIFAR10 images. The codebook is $\\{\pm 1, \pm 1, \ldots\\}$. The naive Pytorch implementation for the sign function could be 

$$ y = x - (x - {\rm torch.sign}(x)).{\rm detach}() $$

such that the derivative is a constant $1.0$. However, there is no stochastic noise to regularize the training. Also, $x$ here can be arbitrarily large. Following the discussion of example 2, we can first map $x$ to a compact set and then take its sign as  

$$
\begin{aligned}
x & \leftarrow x/\sqrt{1+x^2} \\
y & = x - (x - {\rm torch.sign}(x + {\rm noise\\_level}*({\rm torch.rand\\_like}(x) - 0.5))).{\rm detach}()
\end{aligned}
$$

where $0\le {\rm noise\\_level}\le 2$ since $|x|<1$ and we should anneal it to a small enough number when approaching convergence. A less competitive optimizer like Adam benefits a lot from the improved STE. 

![binary_code](https://github.com/lixilinx/ste_revisit/blob/main/binary_code_example.svg)




