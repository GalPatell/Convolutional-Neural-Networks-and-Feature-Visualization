import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Decoder (deconvolutional layers)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x, indices1 = self.pool(x)
        x = F.relu(self.conv2(x))
        x, indices2 = self.pool(x)
        
        # Save the latent representation
        z1 = x
        
        # Decoder
        x = self.unpool(x, indices2)
        x = F.relu(self.deconv2(x))
        x = self.unpool(x, indices1)
        x = F.relu(self.deconv1(x))
        
        # Classification
        z1 = z1.view(-1, 16 * 5 * 5)
        y = F.relu(self.fc1(z1))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        
        return y, x, z1, indices1, indices2

# Function to visualize the latent features
def visualize_latent_features(net, dataloader, device):
    # Accumulate images and their reconstructions for visualization
    original_images = []
    reconstructed_images = []
    channel_images = []

    # Iterate through the dataloader
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass through the network
        net.eval()
        with torch.no_grad():
            outputs, reconstructed, z1, indices1, indices2 = net(images)
        
        # Convert images and reconstructions to numpy arrays
        images_np = images.cpu().numpy()
        rec_np = reconstructed.cpu().numpy()
        
        # Append to lists
        original_images.append(images_np)
        reconstructed_images.append(rec_np)
        
        # Visualize individual channels in the first latent layer
        z1 = z1.cpu().detach().numpy()
        for i in range(z1.shape[1]):
            z1_copy = z1.copy()
            z1_copy[:, :i] = 0
            z1_copy[:, i+1:] = 0
            
            z1_tensor = torch.tensor(z1_copy).to(device)
            z1_tensor = z1_tensor.view(-1, 16, 5, 5)
            
            with torch.no_grad():
                rec_img = net.unpool(z1_tensor, indices2)
                rec_img = F.relu(net.deconv2(rec_img))
                rec_img = net.unpool(rec_img, indices1)
                rec_img = F.relu(net.deconv1(rec_img))
            
            rec_img = rec_img.cpu().detach().numpy()
            channel_images.append(rec_img)
        
        break  # Remove this break if you want to visualize for all batches
    
    # Stack arrays along batch dimension
    original_images = np.vstack(original_images)
    reconstructed_images = np.vstack(reconstructed_images)
    
    # Check if there are channels to visualize
    if len(channel_images) > 0:
        channel_images = np.array(channel_images)
        
        # Plotting all images together
        fig, axs = plt.subplots(len(original_images), 2 + channel_images.shape[0], figsize=(15, 5 * len(original_images)))
        
        for i in range(len(original_images)):
            axs[i, 0].imshow(np.transpose(original_images[i], (1, 2, 0)))
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(np.transpose(reconstructed_images[i], (1, 2, 0)))
            axs[i, 1].set_title('Reconstructed Image')
            axs[i, 1].axis('off')
            
            for j in range(channel_images.shape[0]):
                axs[i, 2 + j].imshow(np.transpose(channel_images[j][i], (1, 2, 0)))
                axs[i, 2 + j].set_title(f'Reconstructed Channel {j+1}')
                axs[i, 2 + j].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # Plotting only original and reconstructed images if no channels to visualize
        fig, axs = plt.subplots(len(original_images), 2, figsize=(10, 5 * len(original_images)))
        
        for i in range(len(original_images)):
            axs[i, 0].imshow(np.transpose(original_images[i], (1, 2, 0)))
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(np.transpose(reconstructed_images[i], (1, 2, 0)))
            axs[i, 1].set_title('Reconstructed Image')
            axs[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    net = Net().to(device)
    net.load_state_dict(torch.load('./cifar_deconv_net.pth'))
    
    # Visualize latent features for a batch of images from the train set
    visualize_latent_features(net, trainloader, device)
    
    # Visualize latent features for a batch of images from the test set
    visualize_latent_features(net, testloader, device)
