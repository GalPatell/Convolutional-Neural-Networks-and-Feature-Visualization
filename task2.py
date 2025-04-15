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
        z = x
        
        # Decoder
        x = self.unpool(x, indices2)
        x = F.relu(self.deconv2(x))
        x = self.unpool(x, indices1)
        x = F.relu(self.deconv1(x))
        
        # Classification
        z = z.view(-1, 16 * 5 * 5)
        y = F.relu(self.fc1(z))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        
        return y, x

# Example function to display images
def imshow(img, labels, predictions=None, classes=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if classes is not None:
        labels_text = ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels)))
        if predictions is not None:
            predictions_text = ' '.join(f'{classes[predictions[j]]:5s}' for j in range(len(predictions)))
            plt.title(f'Ground Truth: {labels_text}\nPredictions: {predictions_text}')
        else:
            plt.title(f'Ground Truth: {labels_text}')
    plt.show()

if __name__ == "__main__":
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

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Instantiate the network
    net = Net()

    # Define the loss functions and optimizer
    criterion_classification = nn.CrossEntropyLoss()
    criterion_reconstruction = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Example training loop
    PATH = './cifar_deconv_net.pth'
    trainModel = True

    if trainModel:
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, reconstructed = net(inputs)
                loss_classification = criterion_classification(outputs, labels)
                loss_reconstruction = criterion_reconstruction(reconstructed, inputs)
                lambda_ = 0.5
                loss = loss_classification + lambda_ * loss_reconstruction  # Combine the two losses
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        torch.save(net.state_dict(), PATH)

    # Example test loop
    net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs, reconstructed = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Generate reconstructed images for visualization if model is not being trained
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs, reconstructed = net(images)

    # Displaying three original vs reconstructed images next to each other in one frame
    fig, axs = plt.subplots(3, 2, figsize=(5, 7.5))  # Create a 3x2 grid of subplots
    for i in range(3):  # Display three images
        img = images[i].cpu().numpy()
        rec_img = reconstructed[i].cpu().detach().numpy()
        
        axs[i, 0].imshow(np.transpose(img, (1, 2, 0)))
        axs[i, 0].set_title('Original')
        axs[i, 0].axis('off')  # Hide axis for better visualization
        
        axs[i, 1].imshow(np.transpose(rec_img, (1, 2, 0)))
        axs[i, 1].set_title('Reconstructed')
        axs[i, 1].axis('off')  # Hide axis for better visualization

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
