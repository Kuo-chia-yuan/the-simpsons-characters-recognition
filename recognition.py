import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import zipfile
import pandas as pd
import torchvision.transforms as T
import matplotlib.pyplot as plt
from google.colab import files
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

!pip install kaggle
files.upload()
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle competitions download -c machine-learning-2023nycu-classification
!unzip machine-learning-2023nycu-classification.zip

transform = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  

    T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
    T.RandomApply([T.RandomVerticalFlip()], p=0.1),
    T.RandomApply([T.RandomRotation(10)], p=0.1),

    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),
])

classes = ('abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson',
           'brandine_spuckler', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler',
           'comic_book_guy', 'disco_stu', 'dolph_starbeam', 'duff_man', 'edna_krabappel', 'fat_tony', 'gary_chalmers',
           'gil', 'groundskeeper_willie', 'homer_simpson', 'jimbo_jones', 'kearney_zzyzwicz', 'kent_brockman', 'krusty_the_clown',
           'lenny_leonard', 'lionel_hutz', 'lisa_simpson', 'lunchlady_doris', 'maggie_simpson', 'marge_simpson', 'martin_prince',
           'mayor_quimby', 'milhouse_van_houten', 'miss_hoover', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann',
           'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier',
           'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'timothy_lovejoy', 'troy_mcclure', 'waylon_smithers')

data_directory = '/content/train/train' 
test_directory = '/content/test-final'

train_data = datasets.ImageFolder(root=data_directory, transform=transform)
test_data = datasets.ImageFolder(root=test_directory, transform=transform)

validation_size = 0.2
num_train_examples = len(train_data)
num_validation = int(validation_size * num_train_examples)
num_train = num_train_examples - num_validation
train_dataset, validation_dataset = random_split(train_data, [num_train, num_validation])

num_epochs = 2
batch_size = 32 
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self, num_classes=50):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

model = CNNModel(num_classes=50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  

losses = []

for epoch in range(num_epochs):
    for batch_idx, (inputs, labels) in enumerate(train_data_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        losses.append(loss.item())

        print(batch_idx, len(train_data_loader), loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total = 0
    correct = 0
    for inputs, labels in train_data_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Epoch {} - Training Accuracy: {}%'.format(epoch + 1, accuracy))

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in valid_data_loader:  
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Validation Accuracy: {}%'.format(accuracy))


model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_data_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for pred in predicted:
          predictions.append(pred.item())

predicted_labels = [classes[pred] for pred in predictions]

plt.plot(losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

file_names = [os.path.basename(test_data.imgs[i][0]) for i in range(len(test_data.imgs))]
df = pd.DataFrame({'id': file_names, 'character': predicted_labels})
df['id'] = df['id'].apply(lambda x: int(x.split('.')[0]))
df = df.sort_values(by='id')

df.to_csv('/content/mysubmission.csv', index=False)