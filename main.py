import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog

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
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_image(image, model, confidence_threshold=0.065):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        confidence = output.item()
        prediction = confidence < confidence_threshold
        return prediction, confidence

class CataractDetectorApp(MDApp):
    def build(self):
        self.model = load_model('model.pth')
        layout = MDBoxLayout(orientation='vertical', padding=10, spacing=10)

        self.title_label = MDLabel(
            text="Cata-Detect\nKnow Your Eyes",
            halign='center',
            theme_text_color='Custom',
            text_color=(0, 0.7, 1, 1),
            font_style='H5'
        )

        self.description_label = MDLabel(
            text="Capture an image using your camera to detect cataracts.",
            halign='center',
            theme_text_color='Secondary',
            font_style='Body1'
        )

        self.camera_button = MDFlatButton(text="Open Camera", pos_hint={"center_x": 0.5})
        self.camera_button.bind(on_release=self.open_camera)

        self.eye_test_button = MDFlatButton(text="Eye Test", pos_hint={"center_x": 0.5})
        self.eye_test_button.bind(on_release=self.eye_test_dialog)

        self.instructions_label = MDLabel(
            text="Instructions:\n1. Click 'Open Camera' to take a photo by pressing the letter 'c'. Make sure to take a high resolution photo with only your face and eyes fully open.\n2. Click 'Eye Test' to start the eye test.\n3. Results will be shown in a pop-up window.",
            theme_text_color='Hint',
            font_style='Body1'
        )

        layout.add_widget(self.title_label)
        layout.add_widget(self.description_label)
        layout.add_widget(self.camera_button)
        layout.add_widget(self.eye_test_button)
        layout.add_widget(self.instructions_label)

        return layout

    def open_camera(self, instance):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Camera')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                image_path = 'captured_image.jpg'
                cv2.imwrite(image_path, frame)
                break

        cap.release()
        cv2.destroyAllWindows()

        self.process_image('captured_image.jpg')

    def crop_image(self, image_path):
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                x_max = int((bboxC.xmin + bboxC.width) * w)
                y_max = int((bboxC.ymin + bboxC.height) * h)

                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                cropped_image = image[y_min:y_max, x_min:x_max]
                return Image.fromarray(cropped_image)

        return None

    def process_image(self, image_path):
        try:
            cropped_image = self.crop_image(image_path)

            if cropped_image is not None:
                prediction, confidence = predict_image(cropped_image, self.model)
                cataract_result = "No Cataracts Detected" if prediction else "Cataracts Detected"
                message = f"{cataract_result}"
            else:
                message = "No face detected in the image."

            self.show_result_popup(message)
        except Exception as e:
            self.show_result_popup(f"An error occurred: {e}")

    def show_result_popup(self, message):
        dialog = MDDialog(
            title="Prediction Result",
            text=message,
            buttons=[
                MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())
            ]
        )
        dialog.open()

    def eye_test_dialog(self, *args):
        self.current_size = 80 
        self.current_round = 1
        self.max_rounds = 5  
        dialog_content = MDBoxLayout(orientation='vertical', padding=20, spacing=20,
                                     size_hint=(0.8, 0.5))

        if self.current_round == 1:
            self.test_dialog_label = MDLabel(
                text="Instructions:\n\n"
                " Please read the letter out loud to asses yourself one arm's length away from the screen.\n"
                ,
                font_style='H5',
                halign='center',
                size_hint_y=None,
                height=self.current_size  
            )
        else:
            self.test_dialog_label = MDLabel(
                text=f"Round {self.current_round}: {self.generate_random_letter()}\n(Size: {self.current_size}px)",
                font_style='H5',
                halign='center',
                size_hint_y=None,
                height=self.current_size  
            )

        dialog_content.add_widget(self.test_dialog_label)

        self.test_dialog = MDDialog(
            title="Eye",
            type="custom",
            content_cls=dialog_content,
            buttons=[
                MDFlatButton(text="Next", on_release=self.next_round),
                MDFlatButton(text="Cancel", on_release=lambda x: self.test_dialog.dismiss())
            ]
        )
        self.test_dialog.open()

    def next_round(self, instance):
        if self.current_round == 1:
            self.current_round += 1
            self.test_dialog_label.text = f"Round {self.current_round}: {self.generate_random_letter()}\n(Size: {self.current_size}px)"
            self.test_dialog_label.font_size = self.current_size
            self.test_dialog_label.height = self.current_size
        elif self.current_round < self.max_rounds:
            self.current_round += 1
            self.current_size = max(10, self.current_size - 15)

            self.test_dialog_label.text = f"Round {self.current_round}: {self.generate_random_letter()}\n(Size: {self.current_size}px)"
            self.test_dialog_label.font_size = self.current_size
            self.test_dialog_label.height = self.current_size
        else:
            self.test_dialog.dismiss()
            self.show_result_popup("Eye test completed!")

    def generate_random_letter(self):
        import random
        import string
        letter = random.choice(string.ascii_uppercase)
        return letter

if __name__ == '__main__':
    CataractDetectorApp().run()
