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


class VQ(torch.nn.Module):
    def __init__(self, num_code, dim_code):
        super(VQ, self).__init__()
        self.codebook = torch.nn.Parameter(torch.randn(num_code, dim_code))
        
    def forward(self, x, noise_level):
        x = torch.nn.functional.softmax(x, dim=1)
        idx = torch.argmax(x - noise_level*torch.rand_like(x), dim=1)
        one_hot = torch.nn.functional.one_hot(idx, num_classes=x.shape[1]).float()
        x = x - (x - one_hot).detach()
        y = x @ self.codebook   
        return y 

    
class Codec(torch.nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        num_code_per_book, dim_code_per_book = 256, 64
        self.codebook1 = VQ(num_code_per_book, dim_code_per_book)
        self.codebook2 = VQ(num_code_per_book, dim_code_per_book)
        self.codebook3 = VQ(num_code_per_book, dim_code_per_book)
        self.codebook4 = VQ(num_code_per_book, dim_code_per_book)
        self.wb1_encoder = torch.nn.Parameter(torch.randn(3*32*32+1, 16384)/(3*32*32)**0.5)
        self.wb2_encoder = torch.nn.Parameter(torch.randn(16384+1, 4*dim_code_per_book)/128)
        self.wb_router = torch.nn.Parameter(torch.randn(4*dim_code_per_book+1, 4*num_code_per_book)/(4*dim_code_per_book)**0.5)
        self.wb1_decoder = torch.nn.Parameter(torch.randn(4*dim_code_per_book+1, 16384)/(4*dim_code_per_book)**0.5)
        self.wb2_decoder = torch.nn.Parameter(torch.randn(16384+1, 3*32*32)/128)
        
    def forward(self, x, noise_level):
        # encoder 
        w, b = self.wb1_encoder[:-1], self.wb1_encoder[-1]
        x = torch.tanh(torch.reshape(x, (-1, 3*32*32)) @ w + b)
        w, b = self.wb2_encoder[:-1], self.wb2_encoder[-1]
        x = torch.tanh(x @ w + b) # we don't VQ x directly 
        
        # router: argmax(x) will point out which code to use
        w, b = self.wb_router[:-1], self.wb_router[-1]
        x = x @ w + b 
        
        # VQ with 4 codebooks 
        x1, x2, x3, x4 = torch.chunk(x, 4, 1)
        x1 = self.codebook1(x1, noise_level)
        x2 = self.codebook1(x2, noise_level)
        x3 = self.codebook1(x3, noise_level)
        x4 = self.codebook1(x4, noise_level)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        # decoder
        w, b = self.wb1_decoder[:-1], self.wb1_decoder[-1]
        x = torch.tanh(x @ w + b)
        w, b = self.wb2_decoder[:-1], self.wb2_decoder[-1]
        x = torch.reshape(x @ w + b, (-1,3,32,32))
        
        return x
    
    
codec = Codec().to(device)
if optimizer.lower() == "psgd":
    opt = KWNS4(codec.parameters(),
                whiten_grad=False, # sparse gradient => whiten momentum  
                lr_params=1e-4/4,
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
    

train_losses, test_losses = [], []
noise_level = 1.0
for epoch in range(100):
    for batch, (data, _) in enumerate(train_loader):
        x = data.to(device)
        xhat = codec(x, noise_level)
        
        opt.zero_grad()
        loss = torch.mean(torch.square(x - xhat))
        loss.backward()
        opt.step()
        train_losses.append(loss.item())
        print(f"epoch {epoch+1}; batch {batch+1}; train loss {train_losses[-1]}")
        
    test_losses.append(test(test_loader))
    print(f"epoch {epoch+1}; test loss {test_losses[-1]}")
    noise_level = 0.9*noise_level + 0.1*0.1 # anneal noise level to 0.1 
    if optimizer.lower() == "psgd":
        # anneal lr_preconditioner to 0.1 and update frequency to 0.01
        opt.param_groups[0]["lr_preconditioner"] *= 0.9
        opt.param_groups[0]["lr_preconditioner"] += 0.1*0.1
        opt.param_groups[0]["preconditioner_update_probability"] *= 0.9
        opt.param_groups[0]["preconditioner_update_probability"] += 0.1*0.01

plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()
smoothed_train_losses = np.convolve(train_losses, np.ones(10)/10)[9:-9]
ax1.plot(-10*np.log10(smoothed_train_losses))
ax1.set_xlabel("Number of iterations")
ax1.set_ylabel("Train PSNR (dB)")
ax2.plot(-10*np.log10(test_losses))
ax2.set_ylabel("Test PSNR (dB)")
ax2.set_xlabel("Number of epochs")
plt.savefig("vq_example.svg")
