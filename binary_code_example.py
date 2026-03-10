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
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=False, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),    
    batch_size=500, shuffle=True)

    
class Codec(torch.nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        self.wb1_encoder = torch.nn.Parameter(torch.randn(3*32*32+1, 16384)/(3*32*32)**0.5)
        self.wb2_encoder = torch.nn.Parameter(torch.randn(16384+1, 1024)/128)
        self.wb1_decoder = torch.nn.Parameter(torch.randn(1024+1, 16384)/32)
        self.wb2_decoder = torch.nn.Parameter(torch.randn(16384+1, 3*32*32)/128)
        
    def forward(self, x, noise_level):
        # encoder 
        w, b = self.wb1_encoder[:-1], self.wb1_encoder[-1]
        x = torch.tanh(torch.reshape(x, (-1, 3*32*32)) @ w + b)
        w, b = self.wb2_encoder[:-1], self.wb2_encoder[-1]
        x = x @ w + b
        
        # tokenization
        x = x * torch.rsqrt(1 + x*x)
        x = x - (x - torch.sign(x + noise_level*(torch.rand_like(x) - 0.5))).detach()
        
        # decoder
        w, b = self.wb1_decoder[:-1], self.wb1_decoder[-1]
        x = torch.tanh(x @ w + b)
        w, b = self.wb2_decoder[:-1], self.wb2_decoder[-1]
        x = torch.reshape(x @ w + b, (-1,3,32,32))
        
        return x
    
    
codec = Codec().to(device)
if optimizer.lower() == "psgd":
    opt = KWNS4(codec.parameters(),
                whiten_grad=True,
                lr_params=1e-4,
                lr_preconditioner=0.5,
                momentum=0.9,
                weight_decay=0)
else:
    opt = torch.optim.Adam(codec.parameters(),
                           lr=1e-4,
                           betas=(0.9, 0.99))
    

def test(data_loader):
    sum_loss, num_samples = 0.0, 0
    with torch.no_grad():
        for inputs, _ in data_loader:
            x = inputs.to(device)
            xhat = codec(x, 0.0)
            sum_loss += torch.sum(torch.square(x - xhat)).item()
            num_samples += x.shape[0]

    return sum_loss / (num_samples*3*32*32)
    

test_losses = []
noise_level = 1.0
for epoch in range(100):
    for batch, (data, _) in enumerate(train_loader):
        x = data.to(device)
        xhat = codec(x, noise_level)
        
        opt.zero_grad()
        loss = torch.mean(torch.square(x - xhat))
        loss.backward()
        opt.step()
        print(f"epoch {epoch+1}; batch {batch+1}; train loss {loss.item()}")
        
    test_losses.append(test(test_loader))
    print(f"epoch {epoch+1}; test loss {test_losses[-1]}")
    noise_level = 0.9*noise_level + 0.1*0.1 # anneal noise level to 0.1 
    if optimizer.lower() == "psgd":
        # anneal lr_preconditioner to 0.1 and update frequency to 0.01
        opt.param_groups[0]["lr_preconditioner"] *= 0.9
        opt.param_groups[0]["lr_preconditioner"] += 0.1*0.1
        opt.param_groups[0]["preconditioner_update_probability"] *= 0.9
        opt.param_groups[0]["preconditioner_update_probability"] += 0.1*0.01

plt.plot(-10*np.log10(test_losses))
plt.ylabel("Test PSNR (dB)")
plt.xlabel("Number of epochs")
plt.title("CIFAR10 reconstruction with binary codes")
plt.savefig("binary_code_example.svg")