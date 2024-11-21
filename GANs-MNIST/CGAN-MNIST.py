import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class Generator(nn.Module):
    # 手写器
    def __init__(self,latent_size, hidden_size, img_size):
        super().__init__()
        self.label_embedding = nn.Embedding(10,10)

        self.model = nn.Sequential(
            nn.Linear(latent_size+10, hidden_size),
            nn.ReLU(), # ???? LeakyRulu & RuLU
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, img_size),
            nn.Tanh()  # ??? 为什么用Tanh
        )

    def forward(self,z,label):
        # z = (batch,latent_size)
        C = self.label_embedding(label).squeeze(1)
        z = torch.cat([z,C],dim=1)
        out = self.model(z)
        return out

class Discriminator(nn.Module):
    # 判别器
    def __init__(self,img_size,hidden_size):
        super().__init__()
        self.label_embedding = nn.Embedding(10,10) # 会把一个[6]拓展维度到二维，之后要squeeze

        self.model = nn.Sequential(
            nn.Linear(img_size+10, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self,z,label):
        # z已经展平为1(batch,784)
        C = self.label_embedding(label)
        C = C.squeeze(1)
        z = torch.cat([z,C],dim=1) # z (batch,784+10)
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
    dataloader = DataLoader(train_data, batch_size=64, num_workers=2, pin_memory=True, shuffle=True)  # 单个数据维度(batch_size,channel,height,width)

    # 提取图片读取img_size
    exp_img = next(iter(dataloader))
    img_size = exp_img[0][0].numel()
    hidden_size = 248
    latent_size = 100
    # 具体训练 实例化、损失函数、优化器定义 -> 具体训练
    d = Discriminator(img_size,hidden_size).to(device)
    g = Generator(latent_size,hidden_size,img_size).to(device)

    loss_fn = nn.BCELoss().to(device)
    d_optim = torch.optim.Adam(d.parameters(),lr=1e-4)
    g_optim = torch.optim.Adam(g.parameters(),lr=1e-4)


    # 开始训练
    g_loss_list = []
    d_loss_list = []
    for epoch in range(10000):
        # 每个epoch的损失之和
        d_loss_all = 0
        g_loss_all = 0
        loader_len = len(dataloader)
        # 计算损失
        for idx, (img,label) in enumerate(dataloader):
            batch_size = img.size(0)

            real_label = torch.ones(batch_size,1).to(device)
            fake_label = torch.zeros(batch_size,1).to(device)
            # img = (batch,img_size)
            img = img.view(batch_size,-1).to(device)
            label = label.view(batch_size,-1).to(device)
            # d_real = (batch,img_size)
            d_real = d(img,label)
            d_real_loss = loss_fn(d_real,real_label)

            # fake_vector = (batch,latent_size)
            fake_vector = torch.randn(batch_size,latent_size).to(device)
            fake_labels = torch.randint(0,10,(batch_size,)).view(batch_size,-1).to(device)
            # fake_img = (batch,img_size)
            fake_img = g(fake_vector, fake_labels)
            d_fake = d(fake_img, fake_labels)
            d_fake_loss = loss_fn(d_fake,fake_label)
            d_optim.zero_grad()
            #计算总的判别器loss
            d_loss = (d_real_loss + d_fake_loss) / batch_size
            # 判别器更新
            d_loss.backward()  # 释放了计算图 - 因此下面必须要再进行一次g，d计算
            d_optim.step()

            g_optim.zero_grad()
            fake_img = g(fake_vector, fake_labels)
            d_fake = d(fake_img, fake_labels)
            g_loss = loss_fn(d_fake,real_label) / batch_size
            # 生成器更新

            g_loss.backward()
            g_optim.step()

            with torch.no_grad():
                d_loss_all += d_loss.item()*batch_size
                g_loss_all += g_loss.item()*batch_size

        with torch.no_grad():
            epoch_d_loss = d_loss_all/loader_len
            epoch_g_loss = g_loss_all/loader_len
            d_loss_list.append(epoch_d_loss)
            g_loss_list.append(epoch_g_loss)
            # if epoch%50 == 0:
            #     print('Discriminator loss:{}'.format(epoch_d_loss))
            #     print('Generator loss:{}'.format(epoch_g_loss))
            # test_ex
            if epoch%2 == 0:
                 print('Discriminator loss:{}'.format(epoch_d_loss))
                 print('Generator loss:{}'.format(epoch_g_loss))
    torch.save(g,'CGAN-g.pth')
    torch.save(d,'CGAN-d.pth')

train()
# 可视化 // 读取模型
device = 'cuda:0' if torch.cuda.is_available else 'cpu'
g = torch.load('CGAN-g.pth')

def digit_img(model,digit):
    z = torch.randn(1,100).to(device)
    label = torch.LongTensor([digit]).to(device)
    img = g(z,label).detach().cpu()
    img = 0.6*img + 0.5
    img = img.view(1,28,28)
    return transforms.ToPILImage()(img)

digit_img(g,8)