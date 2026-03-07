# Passing through gradient with the Dirac delta function
It's common to encounter discontinuous function $y=f(x)$ like quantization, the step and sign function, clustering, etc. There exists many choices to pass through the gradients offering different tradeoffs between bias and variance, say continuous relaxation like Gumbel-Softmax and estimators like REINFORCE and straight-through. But, if the derivative of $f(\cdot)$ can be expressed with the Dirac delta function, we can get unbiased and low variance gradient estimator by passing through the gradient as

$$ E_{v\sim p_V(v)}\left[ \frac{\partial f(x-v)}{\partial x} \right\] $$

where $v$ is drawn from a pdf $p_V(\cdot)$ and $E_v[\cdot]$ is replaced with sample average during training.     

## Recovery of the straight-through estimator (STE) 
Let the target function be $y=f(x)={\rm round}(x)$. Its derivative is $\frac{dy}{dx} = \ldots +\delta(x+1.5) + \delta(x+0.5) + \delta(x-0.5) +\delta(x-1.5)+ \ldots$. Let $v\sim \mathcal{U}(-0.5, 0.5)$, i.e., $p_V(v)=I(-0.5\le v< 0.5)$. Then we have, 

$$ 
\begin{aligned}
E_{v\sim \mathcal{U}(-0.5, 0.5)}\left[ \frac{\partial f(x-v)}{\partial x} \right\] & = \ldots + \int_v \delta(x+0.5-v)I(-0.5\le v< 0.5)dv + \int_v \delta(x-0.5-v)I(-0.5\le v< 0.5)dv + \ldots \\
& = \ldots + I(-0.5\le x+0.5<0.5) + I(-0.5\le x - 0.5 < 0.5) + \ldots \\
& = \ldots + I(-1\le x<0) + I(0\le x < 1) + \ldots = 1
\end{aligned}$$

This is the STE. We can treat the Dirac delta function as a legitimate function and pass through any order of derivatives in this way. Still, we do need noise perturbation to manifest the Dirac delta function as an ordinary function for numerical calculations. 

## Illustration with the Nvidia FP4 quantization function 

Run [this script](https://github.com/lixilinx/Passing-gradient-with-the-Dirac-delta-function/blob/main/fp4_example.py) to generate the following image to illustrate this idea. 
![fp4](https://github.com/lixilinx/Passing-gradient-with-the-Dirac-delta-function/blob/main/fp4_example.svg)

## Example for codec latent variable tokenization 

[This script](https://github.com/lixilinx/Passing-gradient-with-the-Dirac-delta-function/blob/main/binary_code_example.py) gives an example on how to tokenize the latent variable of a simple codec trained for the CIFAR10 images. Without any tricks, the PSGD optimizer can reach peak SNR (PSNR) 30+ dB after a few hundreds of thousands iterations. The STE can only achieve PSNR around 20 dB, far from satisfactory reconstruction.    
![binary_code](https://github.com/lixilinx/Passing-gradient-with-the-Dirac-delta-function/blob/main/binary_code_example.svg)

## Example for weights quantization 
[This script](https://github.com/lixilinx/Passing-gradient-with-the-Dirac-delta-function/blob/main/triple_weight_example.py) gives an example on how to quantize the weights of a simple CIFAR10 codec to $\\{0, \pm 1\\}$. Without any tricks, the PSGD optimizer can reach peak SNR (PSNR) 30+ dB with 100K iterations. Surprisingly, the simple STE can reach 30+ dB PSNR too.    

![triple_weights](https://github.com/lixilinx/Passing-gradient-with-the-Dirac-delta-function/blob/main/triple_weight_example.svg)

## Some comments 
The STE seems to perform pretty well in most cases. Still, it does fail for extreme cases like binary weights or codes. Passing through gradient with the Dirac delta function provides a more principled way to solve such a problem, although it makes the resultant optimization problem more challenging. The selection of $p_V(\cdot)$ can be tricky too, although we just use the normal pdf here for illustration.  

The method here can be generalized to multivariable functions as well, e.g., vector quantization and clustering. Unfortunately, closed-form solutions are available only for very few cases. 

