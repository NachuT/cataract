import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import cv2
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image_np = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        image = Image.fromarray(image_np)
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_image(image_path, model, confidence_threshold=0.4):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        confidence = output.item()
        prediction = (confidence > confidence_threshold)
        print(f"Model output: {output}")
        print(f"Confidence: {confidence:.4f}")
        return prediction, confidence

def main():
    model_path = 'model.pth'
    model = load_model(model_path)
    eye_image_path = '/Users/nachuthenappan/PycharmProjects/pythonProject/Photo on 9-7-24 at 1.32 PM.jpeg'
    prediction, confidence = predict_image(eye_image_path, model)
    if prediction:
        print(f"No, Cataracts detected with confidence {confidence:.4f}")
    else:
        print(f"Yes, Cataracts detected with confidence {confidence:.4f}")
    eye_image = cv2.imread(eye_image_path)
    eye_image_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Eye Image", eye_image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
