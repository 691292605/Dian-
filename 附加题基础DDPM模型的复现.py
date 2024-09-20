import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


tensor_tool = transforms.ToTensor()
transform = (transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
train_dataset = FashionMNIST(root = './data', train = True, transform = transform, download = True)
test_dataset = FashionMNIST(root = './data', train = False, transform = transform, download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1000, shuffle = False)


class New_hidden(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(New_hidden, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i_to_h = nn.Linear(input_size, hidden_size)
        self.h_to_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        combined = self.i_to_h(x) + self.h_to_h(hidden)
        new_hidden = torch.tanh(combined)
        return new_hidden


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.h_to_o = nn.Linear(hidden_size, output_size)
        self.new_hidden = New_hidden(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for data in range(seq_len):
            # 这里对每一个时间步的输入展平为一维向量
            x_flat = x[:, data, :].view(batch_size, -1)
            hidden = self.new_hidden(x_flat, hidden)
        output = self.h_to_o(hidden)
        return output


class DDPM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_time_steps):
        super(DDPM, self).__init__()
        self.rnn = MyRNN(input_size, hidden_size, output_size)
        self.num_time_steps = num_time_steps
        self.beta = torch.linspace(0.0001, 0.02, num_time_steps)  # 预定义扩散过程中的噪声调度
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # 累乘，得到前t步的alpha

    def forward_diffusion(self, x0, t):
        """前向扩散过程: 根据公式 q(x_t | x_0) 生成扩散后的数据"""
        noise = torch.randn_like(x0)
        sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1)
        xt = sqrt_alpha_hat_t * x0 + sqrt_one_minus_alpha_hat_t * noise
        return xt, noise

    def reverse_process(self, xt, t):
        """反向生成过程: 使用RNN预测噪声，生成去噪后的数据"""
        predicted_noise = self.rnn(xt.unsqueeze(1)).squeeze(1)
        predicted_noise = predicted_noise.view(xt.size(0), xt.size(1), -1)# 使用RNN预测噪声
        beta_t = self.beta[t].view(-1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alpha[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1)
        # 根据反向生成过程的公式计算去噪后的数据
        next_x = sqrt_recip_alpha_t.item() * (xt - beta_t.item() * predicted_noise / sqrt_one_minus_alpha_hat_t.item())
        # 添加一个从标准正态分布采样的噪声
        if t > 1:  # 在最后一步t=0不需要加入额外噪声
            sigma_t = torch.sqrt(self.beta[t]).view(-1, 1, 1)
            noise = torch.randn_like(xt)  # 从标准正态分布采样噪声
            next_x = next_x + sigma_t * noise  # 加上噪声项
        return next_x

    def sample(self, shape):
        """采样函数: 反向生成t=0的数据"""
        x_t = torch.randn(shape)  # 从噪声开始采样
        for t in reversed(range(self.num_time_steps)):
            x_t = self.reverse_process(x_t, t)
        return x_t


Input_size = 28 * 28
Hidden_size = 28 * 28
Output_size = 28 * 28
Num_time_steps = 1000
epochs = 10

my_ddpm = DDPM(Input_size, Hidden_size, Output_size, Num_time_steps)
loss_func = nn.MSELoss()
optimizer = optim.Adam(my_ddpm.parameters(), lr = 0.001)
#训练模型，对重构误差进行优化
for epoch in range(epochs):
    for images, _ in train_loader:
        images = images.view(-1, 28, 28)
        T = torch.randint(0, Num_time_steps, (images.size(0),))
        X_t, Noise = my_ddpm.forward_diffusion(images, T)
        Predicted_noise = my_ddpm.rnn(X_t.unsqueeze(1))
        Predicted_noise = Predicted_noise.view(Noise.size(0), Noise.size(1), -1)
        loss = loss_func(Predicted_noise, Noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("第{}轮训练结束!".format(epoch + 1))
    #输出从噪声传回来的图片
    with torch.no_grad():
        generated_images = my_ddpm.sample((64, 28, 28))
        generated_images = generated_images * 0.5 + 0.5
        generated_images = generated_images.clamp(0, 1)
        generated_images = generated_images.view(-1, 1, 28, 28)
        grid_img = torchvision.utils.make_grid(generated_images, nrow=4, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title('Generated Images')
        plt.show()