import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
    
device = torch.device("cuda")
optimizer = "psgd" # psgd or adam

if optimizer.lower() == "psgd":
    try:
        from wrapped_as_torch_optimizer_for_ddp import KWNS4
    except ImportError:
        raise ModuleNotFoundError("Please download psgd.py and wrapped_as_torch_optimizer_for_ddp.py from https://github.com/lixilinx/psgd_torch.")

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),    
    batch_size=500, shuffle=True)


class DTriple(torch.autograd.Function):
    """
    Differentiable version of triple-value function:
        triple(x) = (x>1) - (x<-1)
    which has derivative delta(x+1) + delta(x-1). 
    """
    @staticmethod
    def forward(ctx, input, pdf="gauss", scale=1.0):
        if pdf.lower() == "gauss":
            output = input - scale * torch.randn_like(input)
        else:
            raise ValueError("Unknown pdf")
            
        output = (output>1).to(torch.float32) - (output<-1).to(torch.float32)
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
            grad_input = grad_output * (torch.exp(-(input+1)**2/(2*scale**2)) + 
                                        torch.exp(-(input-1)**2/(2*scale**2)))/(2*torch.pi*scale**2)**0.5
        else:
            raise ValueError("Unknown pdf")
            
        return (grad_input, None, None)

    
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(3*32*32, 16384))
        self.s1 = torch.nn.Parameter(torch.ones(16384)/(3*32*32)**0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(16384))
        self.w2 = torch.nn.Parameter(torch.randn(16384, 1024))
        self.s2 = torch.nn.Parameter(torch.ones(1024)/32)
        self.b2 = torch.nn.Parameter(torch.zeros(1024))
        
    def forward(self, x, pdf, scale):
        weight = self.s1 * DTriple.apply(self.w1, pdf, scale) 
        x = torch.tanh(torch.reshape(x, (-1, 3*32*32)) @ weight + self.b1)
        weight = self.s2 * DTriple.apply(self.w2, pdf, scale) 
        x = x @ weight + self.b2
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(1024, 16384))
        self.s1 = torch.nn.Parameter(torch.ones(16384)/32)
        self.b1 = torch.nn.Parameter(torch.zeros(16384))
        self.w2 = torch.nn.Parameter(torch.randn(16384, 3*32*32))
        self.s2 = torch.nn.Parameter(torch.ones(3*32*32)/(3*32*32)**0.5)
        self.b2 = torch.nn.Parameter(torch.zeros(3*32*32))
        
    def forward(self, x, pdf, scale):
        weight = self.s1 * DTriple.apply(self.w1, pdf, scale) 
        x = torch.tanh(x @ weight + self.b1)
        weight = self.s2 * DTriple.apply(self.w2, pdf, scale) 
        x = x @ weight + self.b2
        return x.reshape((-1, 3, 32, 32))
    
    
encoder = Encoder().to(device)
decoder = Decoder().to(device)
if optimizer.lower() == "psgd":
    opt = KWNS4(list(encoder.parameters()) + list(decoder.parameters()),
                whiten_grad=True,
                lr_params=1e-3,
                lr_preconditioner=0.5,
                momentum=0.9,
                weight_decay=0)
else:
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=1e-3)
    

Losses = []
pdf, scale = "gauss", 1.0
for epoch in range(100):
    for batch, (data, _) in enumerate(train_loader):
        x = data.to(device)
        y = encoder(x, pdf, scale)
        xhat = decoder(y, pdf, scale)
        
        opt.zero_grad()
        loss = torch.mean(torch.square(x - xhat))
        loss.backward()
        opt.step()
        Losses.append(loss.item())
        print(f"epoch {epoch+1}; batch {batch+1}; loss {Losses[-1]}")
       
    scale = 0.9*scale + 0.1*0.1 # anneal noise level to 0.1   
    if optimizer.lower() == "psgd":
        # anneal lr_preconditioner to 0.1 and update frequency to 0.01
        opt.param_groups[0]["lr_preconditioner"] *= 0.9
        opt.param_groups[0]["lr_preconditioner"] += 0.1*0.1
        opt.param_groups[0]["preconditioner_update_probability"] *= 0.9
        opt.param_groups[0]["preconditioner_update_probability"] += 0.1*0.01
        
plt.plot(-10*np.log10(Losses))
plt.ylabel("PSNR (dB)")
plt.xlabel("Number of iterations")
plt.title("Triple weight codec for CIFAR10 reconstruction")
plt.savefig("triple_weight_example.svg")
