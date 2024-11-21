import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class Generator(nn.Module):
    # 生成器
    def __init__(self,latent_size, hidden_size, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, img_size),
            nn.Tanh()
        )

    def forward(self,z):
        # 进来的是潜在空间的向量
        out = self.model(z)
        return out

class Discriminator(nn.Module):
    # 判别器
    def __init__(self,img_size,hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  #
        )

    def forward(self,z):
        # 传入的应该为batch的展平向量
        out = self.model(z)
        return out

def train():
    # 获取数据
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformer = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=0.2, std=0.5)]
    )
    train_data = MNIST("./data", transform = transformer, download=False)
    dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    # 提取图片读取img_size
    exp_img = next(iter(dataloader))
    img_size = exp_img[0][0].numel()
    hidden_size = 248
    latent_size = 100
    # 具体训练 实例化、损失函数、优化器定义 -> 具体训练
    d = Discriminator(img_size,hidden_size)
    g = Generator(latent_size,hidden_size,img_size)

    loss_fn = nn.BCELoss()
    d_optim = torch.optim.Adam(d.parameters(),lr=1e-4)
    g_optim = torch.optim.Adam(g.parameters(),lr=1e-4)

    # 开始训练
    g_loss_list = []
    d_loss_list = []
    for epoch in range(10000):
        d_loss_all = 0
        g_loss_all = 0
        loader_len = len(dataloader)
        # 计算损失
        for idx, (img,_) in enumerate(dataloader):
            batch_size = img.size()[0]
            # img = (batch,img_size)
            img = img.view(batch_size,img_size)
            # d_real = (batch,img_size)
            d_real = d(img)
            d_real_loss = loss_fn(d_real,torch.ones(batch_size,1))

            # fake_vector = (batch,latent_size)
            fake_vector = torch.randn(batch_size,latent_size)
            # fake_img = (batch,img_size)
            fake_img = g(fake_vector)
            d_fake = d(fake_img)
            d_fake_loss = loss_fn(d_fake,torch.zeros(batch_size,1))
            d_optim.zero_grad()
            d_loss = (d_real_loss + d_fake_loss) / batch_size
            # 判别器更新
            d_loss.backward()  # 释放了计算图 - 因此下面必须要再进行一次g，d计算
            d_optim.step()

            g_optim.zero_grad()
            fake_img = g(fake_vector)
            d_fake = d(fake_img)
            g_loss = loss_fn(d_fake,torch.ones(batch_size,1)) / batch_size
            # 生成器更新

            g_loss.backward()
            g_optim.step()

            with torch.no_grad():
                d_loss_all += d_loss*batch_size
                g_loss_all += g_loss*batch_size

        with torch.no_grad():
            epoch_d_loss = d_loss_all/loader_len
            epoch_g_loss = g_loss_all/loader_len
            d_loss_list.append(epoch_d_loss.tolist())
            g_loss_list.append(epoch_g_loss.tolist())
            if epoch%10 == 0:
                print('Discriminator loss:{}'.format(epoch_d_loss))
                print('Generator loss:{}'.format(epoch_g_loss))

train()