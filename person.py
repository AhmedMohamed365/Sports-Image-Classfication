import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import main
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imgSize =(224,224)
mean =[0.4856, 0.5193, 0.4989]
std = [0.3135, 0.2913, 0.3135]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

def plot_prediction(images,predictions):
    #os.mkdir("predictions")
    # mapping = ["Basketball","Football","Rowing","Swimming","Tennis","Yoga" ]
    # index = 0
    #
    # for  img in images:
    #     img = Image.open(f"Test/{img}")
    #     I1 = ImageDraw.Draw(img)
    #
    #     # Add Text to an image
    #     I1.text((28, 36), f"{mapping[prediction[index]]}", fill=(255, 0, 0))
    #     index+=1
    #     # Display edited image
    #     #img.show()
    #     img.save(f"predictions/{index}.jpg")

class sport():
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':

            if ('Basketball') in self.file_list[0]:
                self.label = 0

            if ('Football') in self.file_list[0]:
                self.label = 1

            if ('Rowing') in self.file_list[0]:
                self.label = 2

            if ('Swimming') in self.file_list[0]:
                self.label = 3

            if ('Tennis') in self.file_list[0]:
                self.label = 4

            if ('Yoga') in self.file_list[0]:
                self.label = 5

        else:
            self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), 0


test_transform = transforms.Compose([
    # left, top, right, bottom
    transforms.Resize(imgSize),
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor(mean), torch.tensor(std))
])



Testimages_name = os.listdir("Test")
testdir = "Test"
testData = sport(Testimages_name, testdir, mode="test", transform=test_transform)
FinaltestLoader = DataLoader(testData, batch_size=32, shuffle=False)

# output prediction
model = main.model


Final_prediction = []
model.eval()
model.to(device)
with torch.no_grad():
    for idx, (images, labels) in enumerate(FinaltestLoader):
        images = images.to(device=device)
        labels = labels.to(device=device)
        outputs = model(images)
        _, preds = outputs.max(1)
        Final_prediction.extend(preds)

prediction = []
for i in Final_prediction:
    prediction.append(i.item())

images_name = os.listdir(testdir)

# output to datafRame



rawData = {"image_name": images_name, "label": prediction}

Data = pd.DataFrame(rawData)

Data = Data.sort_values("image_name")
plot_prediction(images_name,Data["label"])
# output submission file


Data.to_csv("test.csv", index=False)