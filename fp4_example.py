import matplotlib.pyplot as plt
import numpy as np

# nvidia fp4 has values {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}

def fp4(x):
    """
    Simulate the fp4 quantization function. 
    """
    s = np.sign(x)
    x = np.abs(x)
    if x > (1.5 + 2)/2:
        if x > (3 + 4)/2:
            if x > (4 + 6)/2:
                y = 6
            else:
                y = 4
        else:
            if x > (2 + 3)/2:
                y = 3
            else:
                y = 2
    else:
        if x > (0.5 + 1)/2:
            if x > (1 + 1.5)/2:
                y = 1.5
            else:
                y = 1
        else:
            if x > (0 + 0.5)/2:
                y = 0.5
            else:
                y = 0 
    return s * y


x = np.arange(-8, 8, 0.001)
y = np.stack([fp4(a) for a in x])
plt.plot(x, y, 'k')

dy_dx = np.zeros_like(x) # Dirac delta function 
for i in range(1, len(x)):
    dy_dx[i] = y[i] - y[i-1] 
plt.plot(x, dy_dx, linewidth=0.5, color='red')
plt.scatter(x[dy_dx>0], dy_dx[dy_dx>0], marker='^', linewidth=0.5, color="red", label="_nolegend_")

sigma2 = 0.5
manifested_dy_dx = np.zeros_like(x)
for i in range(len(dy_dx)):
    if dy_dx[i] > 0:
        manifested_dy_dx += dy_dx[i] * np.exp(-(x - x[i])**2/(2*sigma2))/(2*np.pi*sigma2)
plt.plot(x, manifested_dy_dx, color="blue")

plt.xlabel(r"$x$")
plt.legend([r"${\rm fp4}(x)$", r"$\frac{d\, {\rm fp4}(x)}{dx}$", r"$E_{v\sim \mathcal{N}(0, 0.5)}\left[\frac{d\, {\rm fp4}(x-v)}{dx}\right]$"])
plt.savefig("fp4_example.svg")             
