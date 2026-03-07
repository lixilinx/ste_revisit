# Passing through gradient with the Dirac delta function
It's common to encounter discontinuous function $y=f(x)$ like quantization, step and sign functions, clustering, etc. There exists many choices to pass through the gradients to offer different tradeoffs between bias and variance, say continuous relaxation like Gumbel-Softmax and estimators like REINFORCE and straight-through. But, if the derivative of $f(\cdot)$ can be expressed with the Dirac delta function, generally we can get less biased and lower variance gradient estimator by passing through the gradient as

$$ E_{v\sim p_V(v)}\left[ \frac{\partial f(x-v)}{\partial x} \right\] $$

where $v$ is drawn from a pdf $p_V(\cdot)$ and $E_v[\cdot]$ is replaced with sample average during training.     

## Recovery of the straight-through estimator (STE) 
Let the target function be $y=f(x)={\rm round}(x)$. Its derivative is $\frac{dy}{dx} = \ldots +\delta(x+1.5) + \delta(x+0.5) + \delta(x-0.5) +\delta(x-1.5)+ \ldots$. Let $v\sim \mathcal{U}(-0.5, 0.5)$, i.e., $p_V(v)=I(-0.5\le v< 0.5)$. Then, we have, 

$$ 
\begin{aligned}
E_{v\sim \mathcal{U}(-0.5, 0.5)}\left[ \frac{\partial f(x-v)}{\partial x} \right\] & = \ldots + \int_v \delta(x+0.5-v)I(-0.5\le v< 0.5)dv + \int_v \delta(x-0.5-v)I(-0.5\le v< 0.5)dv + \ldots \\
& = \ldots + I(-0.5\le x+0.5<0.5) + I(-0.5\le x - 0.5 < 0.5) + \ldots \\
& = \ldots + I(-1\le x<0) + I(0\le x < 1) + \ldots = 1
\end{aligned}$$

This is the STE. The injected noise reduces the bias of STE. We can treat the Dirac delta function as a legitimate function and pass through any order of derivatives in this way. Noise injection is to manifest the Dirac delta function as an ordinary function for numerical calculations. Clearly, with $p_V(v)=\delta(v)$, we recover the original derivative.     

## Concept illustration with the Nvidia FP4 quantization function 

Run [this script](https://github.com/lixilinx/Passing-through-gradient-with-the-Dirac-delta-function/blob/main/fp4_example.py) to generate the following image to illustrate this idea. 
![fp4](https://github.com/lixilinx/Passing-through-gradient-with-the-Dirac-delta-function/blob/main/fp4_example.svg)

## Example for encoder latent variable tokenization 

[This script](https://github.com/lixilinx/Passing-through-gradient-with-the-Dirac-delta-function/blob/main/binary_code_example.py) gives an example on how to tokenize the latent variable of a simple codec trained for the CIFAR10 images. The codebook is $\\{\pm 1, \pm 1, \ldots\\}$. The target function and its derivative are $f(x)={\rm sign}(x)$ and $2\delta(x)$, respectively. Without any tricks, the PSGD optimizer can reach peak SNR (PSNR) 30+ dB after a few hundreds of thousands iterations. The STE can only achieve PSNR 20+ dB, generating unsatisfactory reconstructed images.    
![binary_code](https://github.com/lixilinx/Passing-through-gradient-with-the-Dirac-delta-function/blob/main/binary_code_example.svg)

## Example for weights quantization 
[This script](https://github.com/lixilinx/Passing-through-gradient-with-the-Dirac-delta-function/blob/main/triple_weight_example.py) gives an example on how to quantize the weights of a simple CIFAR10 codec to $\\{0, \pm 1\\}$, up to certain scaling differences. The target function and its derivative are $f(x)=I(x>1) - I(x<-1)$ and $\delta(x+1) + \delta(x-1)$, respectively. Since $f(x)$ is not scaling-invariant, we do need to anneal down the noise level during training. The PSGD optimizer can reach PSNR 30+ dB with 100K iterations. Surprisingly, the simple STE can reach PSNR 30+ dB too.    

![triple_weights](https://github.com/lixilinx/Passing-through-gradient-with-the-Dirac-delta-function/blob/main/triple_weight_example.svg)

## Concluding comments 
The classic STE seems to perform pretty well in many cases. Still, its bias clearly limits its performance for certain applications. Passing through gradient with the Dirac delta function provides a more principled way to solve such problems, although it makes the resultant optimization problem more challenging. The selection of $p_V(\cdot)$ can be tricky too, although here we just use the normal pdf for illustration.  

The method here resembles STE and Gumbel-Softmax like methods that all produce low variance gradient estimators. But, we use noise injection to reduce the bias. It's also related to stochastic rounding for bias reduction. But, stochastic rounding does not connect the pdf of injected noise and gradient propagation. 

This Dirac delta function method can be generalized to multivariable functions as well, e.g., vector quantization and clustering. Unfortunately, closed-form solutions are available only for very few cases (to my knowledge). 

