import matplotlib.pyplot as plt
import numpy as np 
import torch
from torchvision import datasets, transforms
    
device = torch.device("cuda")
optimizer = "psgd" # psgd or adam
passing_derivative_with = "dirac_delta" # dirac_delta or straight_through 

if optimizer.lower() == "psgd":
    try:
        from wrapped_as_torch_optimizer_for_ddp import KWNS4
    except ImportError:
        raise ModuleNotFoundError("Please download psgd.py and wrapped_as_torch_optimizer_for_ddp.py from https://github.com/lixilinx/psgd_torch.")

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),    
    batch_size=500, shuffle=True)


class DSign(torch.autograd.Function):
    """
    A differentiable version of the sign function, whose derivative is 2*delta(x). 
    """
    @staticmethod
    def forward(ctx, input, pdf="gauss", scale=1.0):
        if pdf.lower() == "gauss":
            output = input - scale * torch.randn_like(input)
        else:
            raise ValueError("Unknown pdf")
            
        ctx.save_for_backward(input)
        ctx.pdf = pdf
        ctx.scale = scale 
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        pdf = ctx.pdf
        scale = ctx.scale 
        
        if pdf.lower() == "gauss":
            grad_input = grad_output * 2*torch.exp(-input*input/(2*scale**2))/(2*torch.pi*scale**2)**0.5
        else:
            raise ValueError("Unknown pdf")
            
        return (grad_input, None, None)
    
    
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.wb1 = torch.nn.Parameter(torch.randn(3*32*32+1, 16384)/(3*32*32)**0.5)
        self.wb2 = torch.nn.Parameter(torch.randn(16384+1, 1024)/128)
        
    def forward(self, x):
        w, b = self.wb1[:-1], self.wb1[-1]
        x = torch.tanh(torch.reshape(x, (-1, 3*32*32)) @ w + b)
        w, b = self.wb2[:-1], self.wb2[-1]
        x = x @ w + b
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.wb1 = torch.nn.Parameter(torch.randn(1024+1, 16384)/32)
        self.wb2 = torch.nn.Parameter(torch.randn(16384+1, 3*32*32)/128)
        
    def forward(self, x):
        w, b = self.wb1[:-1], self.wb1[-1]
        x = torch.tanh(x @ w + b)
        w, b = self.wb2[:-1], self.wb2[-1]
        x = torch.reshape(x @ w + b, (-1,3,32,32))
        return x
    
    
encoder = Encoder().to(device)
decoder = Decoder().to(device)
if optimizer.lower() == "psgd":
    opt = KWNS4(list(encoder.parameters()) + list(decoder.parameters()),
                whiten_grad=True,
                lr_params=2e-5,
                lr_preconditioner=0.5,
                momentum=0.9,
                weight_decay=0)
else:
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=2e-5)
    

Losses = []
for epoch in range(100):
    for batch, (data, _) in enumerate(train_loader):
        x = data.to(device)
        y = encoder(x)
        if passing_derivative_with == "dirac_delta":
            z = DSign.apply(y)
        else: # straight_through
            z = y + (torch.sign(y) - y).detach()
        xhat = decoder(z)
        
        opt.zero_grad()
        loss = torch.mean(torch.square(x - xhat))
        loss.backward()
        opt.step()
        Losses.append(loss.item())
        print(f"epoch {epoch+1}; batch {batch+1}; loss {Losses[-1]}")
        
    if optimizer.lower() == "psgd":
        # anneal lr_preconditioner to 0.1 and update frequency to 0.01
        opt.param_groups[0]["lr_preconditioner"] *= 0.9
        opt.param_groups[0]["lr_preconditioner"] += 0.1*0.1
        opt.param_groups[0]["preconditioner_update_probability"] *= 0.9
        opt.param_groups[0]["preconditioner_update_probability"] += 0.1*0.01

plt.plot(-10*np.log10(Losses))
plt.ylabel("PSNR (dB)")
plt.xlabel("Number of iterations")
plt.title("CIFAR10 reconstruction with binary codes")
plt.savefig("binary_code_example.svg")