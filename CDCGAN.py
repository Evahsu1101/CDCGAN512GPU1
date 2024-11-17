# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 06:01:38 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 00:55:14 2023

@author: User
"""

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision.models import inception_v3
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    
# 數據集定義
class CustomDataset(Dataset):
    # 初始化
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform

        self.image_files = []
        self.labels = []

        # 加載數據和標籤
        for label in os.listdir(root_folder):
            label_path = os.path.join(root_folder, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    self.image_files.append(os.path.join(label_path, image_file))
                    self.labels.append(int(label))  # 文件夾名稱為0 OR 1

    # 獲取數據及長度
    def __len__(self):
        return len(self.image_files)

    # 獲取數據項
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# 圖像預處理256*256
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  
    transforms.Resize((512, 512)),# 調整畫素為1024x1024
    transforms.ToTensor(), 
])


# 定義路徑
root_folder = r"./classfication"
output_folder = r"./generation_test"
output_folder1 = r"./classification_generation/0"
output_folder2 = r"./classification_generation/1"
# root_folder = "/Users/xucixin/Desktop/陳漢興學長程式碼/CDCGAN/分類"
# output_folder = "/Users/xucixin/Desktop/陳漢興學長程式碼/CDCGAN/測試生成"
# output_folder1 = "/Users/xucixin/Desktop/陳漢興學長程式碼/CDCGAN/分類生成/0"
# output_folder2 = "/Users/xucixin/Desktop/陳漢興學長程式碼/CDCGAN/分類生成/1"
custom_dataset = CustomDataset(root_folder, transform=transform)

# dataloader
# batch_size = 43
batch_size = 86
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)



class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, img_channels, img_size):
        super(Generator, self).__init__()
        # Initialization
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Calculate size of the input to the first transposed convolution layer
        self.init_size = img_size // 4  # This can be adjusted depending on the desired architecture ＃計算進入轉置卷積層前的特徵圖大小。這裡使用的是 img_size // 4，代表圖像尺寸會縮小 4 倍。
        self.l1 = nn.Sequential(nn.Linear(latent_dim + condition_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, conditions):
        # Directly use conditions as they are no longer embedded
        gen_input = torch.cat((noise, conditions), dim=1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, condition_dim, img_channels, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels + condition_dim, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(512 * (img_size // 16) ** 2, 1024),
            nn.Linear(512 * (img_size // 16) ** 2, 1024),
            # nn.BatchNorm1d(1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            # nn.Linear(1024, 1),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, conditions):
        # Expand conditions to match the image spatial dimensions
        conditions = conditions.view(conditions.size(0), conditions.size(1), 1, 1)
        conditions = conditions.expand(conditions.size(0), conditions.size(1), img.size(2), img.size(3))
        img_with_conditions = torch.cat((img, conditions), dim=1)
        
        conv_out = self.conv_layers(img_with_conditions)
        fc1_out = self.fc1(conv_out)
        validity = self.fc2(fc1_out)
        
        return validity

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")   



# 超參數設置
latent_size = 200
latent_dim = 200
condition_dim = 1
img_channels = 3
# img_size = 256
img_size = 512
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 創建及初始化生成器及判別器
generator = Generator(latent_dim, condition_dim, img_channels, img_size).to(device)
discriminator = Discriminator(condition_dim, img_channels, img_size).to(device)

# 定義損失函數和優化器
criterion = nn.BCELoss().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))



def plot_generator_loss(generator_losses, title):
    plt.figure(figsize=(12, 8))
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


    
    
model_save_dir= r"./model"
# 初始化紀錄的列表
overall_accuracies = []
class_0_accuracies = []
class_1_accuracies = []
generator_losses = []
discriminator_losses = []
is_scores_list = []  
fid_scores_list = [] 
lpips_scores = []  

#訓練參數
num_classes = 2
epochs = 10000
sample_interval = 100  # 10個EPOCH生成一張圖
num_images_to_generate = 100
sample_interval1 = 100
#訓練迴圈
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(data_loader):
        real_images = images.to(device)

        conditions = labels.to(device)
        conditions = labels.unsqueeze(1).to(device)

        # 將判別器梯度清為0，以便新一輪的反向傳播和優化
        optimizer_D.zero_grad()

        #生成假圖片
        z = torch.randn(batch_size, latent_size).to(device) # 生成隨機的噪音張量(10,106)
        fake_images = generator(z, conditions)#將隨機噪音z和條件傳回GENERATOR 生成假圖片
        inputdata = torch.cat((z, conditions), dim=1)
        # 判別器對於真實圖像的輸出
        real_outputs = discriminator(real_images, conditions)#判別器對於Real_images進行評估conditions一起放進去輔助評估
        real_targets = torch.ones_like(real_outputs)#將Real_outputs鑑別器的輸出為1

        # 判別器對於假圖像的輸出
        fake_outputs = discriminator(fake_images.detach(), conditions)#判別器對於fake_images進行評估conditions一起放進去輔助評估
        fake_targets = torch.zeros_like(fake_outputs)#將fake_outputs鑑別器的輸出為0

        #判別器的損失函數
        loss_real = criterion(real_outputs, real_targets)#判別器對於真實圖圖像的輸出與真實標籤的差異
        loss_fake = criterion(fake_outputs, fake_targets)#判別器對於假圖像的輸出與假標籤的差異
        loss_D = loss_real + loss_fake #總LOSS判別器
        #使判別器更好區分真實及生成圖像
        loss_D.backward()
        optimizer_D.step()#更新判別器的參數

        #將生成器梯度清為0，以便新一輪的反向傳播和優化
        optimizer_G.zero_grad()

        # 計算假圖像的判別結果
        fake_outputs = discriminator(fake_images, conditions)
        real_targets = torch.ones_like(fake_outputs)  # 生成器希望生成的圖被判別器定義為真實的

        # 生成器的損失
        loss_G = criterion(fake_outputs, real_targets)

        loss_G.backward()
        optimizer_G.step()
        generator_losses.append(loss_G.item())
        discriminator_losses.append(loss_D.item())
        # 每個EPOCH計算準確率
       # 每个epoch的信息输出
        print(f'Epoch [{epoch+1}/{epochs}], Step [{epoch+1}/{len(data_loader)}], '
                  f'Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')


        # 保存生成器的生成的圖像和標籤
        if epoch % sample_interval == 0:
            with torch.no_grad():
             # 將生成的圖像和標籤丟入變量中
               samples_with_labels = [{'image': fake_images[i], 'label': conditions[i].cpu().numpy()} for i in range(batch_size)]
               save_image(fake_images.data[:1], f"{output_folder}/Test5epoch_{epoch}.png", nrow=1, normalize=True)
               
 # 每1000个Epoch保存一次模型
        if (epoch + 1) % 100 == 0:
            generator_save_path = os.path.join(model_save_dir, f'CDCGAN T2 generator_epoch_{epoch+1}.pth')

        
            torch.save(generator.state_dict(), generator_save_path)
            
            print(f"Models saved at epoch {epoch+1}")
# 定義保存模型資料夾

save_folder = r"./model"
os.makedirs(save_folder, exist_ok=True)  # 确保文件夹存在

# 定義保存模型路徑
generator_path = os.path.join(save_folder, 'CDCGAN T2 generator EPOCH10000.pth')
#discriminator_path = os.path.join(save_folder, 'CWGAN-GP T2 discriminator EPOCH10000.pth')

# 保存模型
torch.save(generator.state_dict(), generator_path)


print(f"Generator model saved at {generator_path}")

              
                
#訓練結束畫圖      
plot_generator_loss(generator_losses, 'CDCGAN Training Losses')


# 保存模型參數
torch.save(generator.state_dict(), 'CDCGAN T2 generator EPOCH10000.pth')
torch.save(discriminator.state_dict(), 'CDCGAN T2 discriminator EPOCH10000.pth')



import os
import torch
from torchvision.utils import save_image

def generate_and_save_images(generator, category, num_images, latent_dim, output_dir, device):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 设置生成器为评估模式
    generator.eval()
    # 创建潜在空间的噪声向量
    noise = torch.randn(num_images, latent_dim, device=device)
    # 创建类别标签
    labels = torch.full((num_images, 1), category, device=device, dtype=torch.float32)  
    # 生成图片
    with torch.no_grad():
        generated_images = generator(noise, labels).detach().cpu()
    # 保存图片
    for i, img in enumerate(generated_images):
        save_image(img, os.path.join(output_dir, f'category_{category}_image_{i}.png'))


latent_dim = 200  


output_dir = r"./generation"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 加载生成器模型
# generator = Generator(latent_dim, condition_dim=1, img_channels=3, img_size=256).to(device)
generator = Generator(latent_dim, condition_dim=1, img_channels=3, img_size=512).to(device)
generator.load_state_dict(torch.load(r"./model/CWGAN-GP C T2 generator_epoch_5000.pth", map_location=device))


generate_and_save_images(generator, category=0, num_images=100, latent_dim=latent_dim, output_dir=output_dir, device=device)

generate_and_save_images(generator, category=1, num_images=100, latent_dim=latent_dim, output_dir=output_dir, device=device)
