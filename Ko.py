
label = torch.full((batch_size, 1, 1, 1), 1, device=device)


output = output.view(batch_size, -1)


label = label.float()

output = output.float()

# Atualiza o Discriminador com dados reais e gerados
netD.zero_grad()
real = data[0].to(device)
batch_size = real.size(0)
label = torch.full((batch_size, 1, 1, 1), 1, device=device).float()  # Convertendo para float

output = netD(real)
errD_real = criterion(output, label)
errD_real.backward()

noise = torch.randn(batch_size, nz, 1, 1, device=device)
fake = netG(noise)
label.fill_(0)
output = netD(fake.detach())
errD_fake = criterion(output, label)
errD_fake.backward()
optimizerD.step()

# Atualiza o Gerador
netG.zero_grad()
label.fill_(1)
output = netD(fake)
errG = criterion(output, label)
errG.backward()
optimizerG.step()

if i % 100 == 0:
    print(f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} \
          Loss D: {errD_real+errD_fake}, Loss G: {errG}')
    save_image(fake.data[:25], f'images/{epoch}_{i}.png', nrow=5, normalize=True)

    noise = torch.randn(batch_size, nz, 1, 1, device=device)


    fake = netG(noise)



    save_image(fake.data[:25], 'nome_do_arquivo.png', nrow=5, normalize=True)



    import torch

# Definir o dispositivo (GPU se disponível, caso contrário CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tamanho do vetor de entrada para o gerador (nz)
nz = 100

# Gerar ruído aleatório
noise = torch.randn(1, nz, 1, 1, device=device)

# Passar pelo gerador para gerar uma imagem
fake = netG(noise)

# Salvar a imagem gerada
save_image(fake.data, 'imagem_gerada.png')


import torch
from torchvision.utils import save_image

# Definir o dispositivo (GPU se disponível, caso contrário CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tamanho do vetor de entrada para o gerador (nz)
nz = 100

# Defina seu gerador netG aqui
# Exemplo:
# netG = Generator().to(device)

# Gerar ruído aleatório
noise = torch.randn(1, nz, 1, 1, device=device)

# Passar pelo gerador para gerar uma imagem
fake = netG(noise)

# Salvar a imagem gerada
save_image(fake.detach().cpu(), 'imagem_gerada.png', normalize=True)



















import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Hiperparâmetros
batch_size = 64
lr = 0.0002
epochs = 100
nz = 100  # Tamanho do vetor de entrada para o gerador

# Transformações e carregamento do conjunto de dados
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Definindo o Gerador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Definindo o Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Inicializando modelos, critérios e otimizadores
netG = Generator().cuda()
netD = Discriminator().cuda()

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# Treinamento
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # Atualiza o Discriminador com dados reais e gerados
        netD.zero_grad()
        real = data[0].cuda()
        batch_size = real.size(0)
        label = torch.full((batch_size,), 1, device='cuda')
        
        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        noise = torch.randn(batch_size, nz, 1, 1, device='cuda')
        fake = netG(noise)
        label.fill_(0)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()
        
        # Atualiza o Gerador
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} \
                  Loss D: {errD_real+errD_fake}, Loss G: {errG}')
            save_image(fake.data[:25], f'images/{epoch}_{i}.png', nrow=5, normalize=True)
