# DCGAN
import torch.nn as nn
import torch
import torchvision.transforms as transformer
import torchvision
import matplotlib.pyplot as plt



# 假设整个模型的输入是MNIST(64,1,)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # model 是转置卷积过程 从(10,10) 到(28,28)的过程
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # 进来的是一个向量 latent_vector (N,16)
        z = torch.unsqueeze(z, 1)  # (N,1,16)
        z = z.view(z.size(0), 1, 4, 4)
        outputs = self.model(z)
        return outputs


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 图像为(N,1,28,28)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=2, stride=2, padding=1)  # (N,1,2,2)
        )
        self.fc1 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # 进来的是图像
        x = self.model(z)
        x = x.view(x.size(0), -1)
        output = self.sigmoid(self.fc1(x))
        return output


# 图像数据的处理
transformers = transformer.Compose(
    [transformer.ToTensor(),
     transformer.Normalize(mean=0.2, std=0.5)]
)

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=False, transform=transformers)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=2, pin_memory=True,
                                               shuffle=True)


def train():  # 测试模型的运行过程
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 模型实例化和损失函数、优化器
    d = Discriminator().to(device)
    g = Generator().to(device)

    loss = nn.BCELoss()
    g_optim = torch.optim.Adam(g.parameters(), lr=1e-4)
    d_optim = torch.optim.Adam(d.parameters(), lr=1e-4)

    epoch_g_loss = []
    epoch_d_loss = []
    batch_g_loss = []
    batch_d_loss = []
    for epoch in range(10):  # 记录每个epoch的d_loss和g_loss , 并在一定次数后显示
        d_loss_sum = 0
        g_loss_sum = 0
        d_temp = []
        g_temp = []
        for idx, (imgs, _) in enumerate(train_dataloader):  # img - (N,1,28,28)
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            output = d(imgs)
            real_loss = loss(output, real_labels)
            # 随机抽取向量
            latent_z = torch.randn(batch_size, 16).to(device)
            output = g(latent_z)
            output = d(output)
            fake_loss = loss(output, fake_labels)
            d_loss = real_loss + fake_loss
            # 更新判别器
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # 训练生成器,要重建计算图
            output = g(latent_z)
            output = d(output)
            g_loss = loss(output, real_labels)
            # 更新生成器
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            with torch.no_grad():
                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()
                # 记录每个batch的loss值
                d_temp.append(d_loss.item())
                g_temp.append(g_loss.item())
                if epoch % 2 == 0 and idx % 300 == 0:
                    print(f'[{epoch}, {idx}] Generator loss: {g_loss_sum}, Discriminator loss: {d_loss_sum}')
        # 记录batch的值
        batch_g_loss.append(g_temp)
        batch_d_loss.append(d_temp)
        # 记录epoch的值
        epoch_d_loss.append(d_loss_sum)
        epoch_g_loss.append(g_loss_sum)

    return (batch_g_loss, batch_d_loss), (epoch_g_loss, epoch_d_loss), idx


batch_loss, epoch_loss, idx = train()  # 画两种图1:用与展示10个epoch中，loss随着batch的下降 2:10个epoch的总的loss的下降
print('Train Done')


# 3张图,10epoch的g和dloss对比，10种batch下的g_loss和10种batch下的d_loss
epoch = [i for i in range(10)]
fig = plt.figure(figsize=(20, 18), dpi=100)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
# batch_loss (batch_g_loss,batch_d_loss) batch_g_loss-(epoch,batch,loss)
# batch_g_loss G的10个epoch中，batch为横坐标，loss为纵坐标

for i in range(10):
    ax1.plot(range(idx + 1), batch_loss[0][i], label=f'Epoch{i}')

ax1.set_title('G_loss within batch')
ax1.set_xlabel('Batch')
ax1.set_ylabel('Loss')
ax1.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

for i in range(10):
    ax2.plot(range(idx + 1), batch_loss[1][i], label=f'Epoch{i}')

ax2.set_title('D_loss within batch')
ax2.set_xlabel('Batch')
ax2.set_ylabel('Loss')
ax2.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

# Epoch loss

ax3.plot(range(10), epoch_loss[0], label='G_loss')
ax3.plot(range(10), epoch_loss[1], label='D_loss')
ax3.set_title('D&G loss within Epoch')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

plt.show()