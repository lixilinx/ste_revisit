import torch 

def fp4(x, noise_level):
    """
    Simulation of the Nvidia FP4 quantization function that supports the STE.
    The FP4 has values {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    Set the noise level in range 0 <= noise_level <= 1. 
    """
    with torch.no_grad():
        s = torch.sign(x)
        a = torch.abs(x)
        z = torch.min(2*a, torch.min(a + 2, 0.5*a + 4))
        z -= noise_level * (torch.rand_like(z) - 0.5)
        z = torch.round(z)
        z = torch.clamp(z, min=-7, max=7)
        y = s * torch.max(0.5*z, torch.max(z - 2, 2*z - 8))
    y = x - (x - y).detach()
    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x = torch.arange(-6, 6, 0.001)
    y = fp4(x, noise_level=0.5)
    plt.plot(x, y)
    
    x = torch.arange(-6, 6, 0.01)
    y = fp4(x, noise_level=0.0)
    plt.plot(x, y)
    plt.grid()
    plt.xlabel(r"$x$")
    plt.ylabel(r"Sample of ${\rm fp4}(x - v)$")
    plt.legend([r"$v\sim U(-0.25, 0.25)$", r"$v=0$"])
    plt.savefig("fp4_example.svg")
    
