git remoto adicionar origem https://github.com/Coelho7kook/Ia.git
 git branch -M main 
git push -u origin main

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


