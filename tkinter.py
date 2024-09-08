import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 9 * 9, 512)  # Adjust based on input size
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

# Load the trained model
def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the transformation for input images
def preprocess_image(image):
    # Convert image to RGB
    image = image.convert('RGB')

    # Define transformations for resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Mean and std dev for normalization
    ])

    image = transform(image).unsqueeze(0)
    return image

# Predict function with debug outputs
def predict_image(image, model, confidence_threshold=0.4):
    # Preprocess the image
    image = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        confidence = output.item()
        prediction = (confidence > confidence_threshold)
        return prediction

# Function to open file and predict
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Show loading animation
            show_loading_animation()

            image = Image.open(file_path)
            image.thumbnail((250, 250))  # Resize image for processing

            # Predict the class of the image
            prediction = predict_image(image, model)
            if prediction:
                messagebox.showinfo("Prediction Result", "No Cataracts Detected\nGood news! Your eyes appear healthy.")
            else:
                messagebox.showwarning("Prediction Result", "Cataracts Detected\nPlease consult a healthcare professional for further examination.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            hide_loading_animation()

# Show loading animation
def show_loading_animation():
    animation_label.config(text="Processing...", font=("Comfortaa", int(36 * scaling_factor), "bold"), fg="#00BFFF")

# Hide loading animation
def hide_loading_animation():
    animation_label.config(text="")

# Create a custom button widget
class CustomButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(font=("Comfortaa", int(24 * scaling_factor), "bold"), bg="#1E90FF", fg="white", relief=tk.RAISED,
                    borderwidth=int(3 * scaling_factor), padx=int(20 * scaling_factor), pady=int(10 * scaling_factor))
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def on_enter(self, e):
        self.config(bg="#4682B4")

    def on_leave(self, e):
        self.config(bg="#1E90FF")

    def on_click(self, e):
        self.config(bg="#0056b3")
        root.after(100, self.on_leave, e)

# Create the main Tkinter window
root = tk.Tk()
root.title("Cataracts Detector")

# Scaling factor
scaling_factor = 1

# Set window size and frame padding based on scaling factor
window_width = int(600 * scaling_factor)
window_height = int(700 * scaling_factor)
root.geometry(f"{window_width}x{window_height}")
root.configure(bg="#1C1C1C")

# Create a canvas for background gradient
canvas = tk.Canvas(root, width=window_width, height=window_height, bg="#1C1C1C", highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

# Draw a light gradient on the canvas
def draw_gradient(canvas, width, height):
    for i in range(height):
        color = f'#{255:02x}{255:02x}{255:02x}'
        canvas.create_line(0, i, width, i, fill=color, tags="gradient")

draw_gradient(canvas, window_width, window_height)

# Create a main frame on top of the canvas
main_frame = tk.Frame(root, bg="#1C1C1C")
main_frame.place(relwidth=1, relheight=1)

# Create a title label with typing animation
def typing_animation(label, text, delay=100):
    def update_text(i=0):
        if i < len(text):
            label.config(text=text[:i + 1])
            root.after(delay, update_text, i + 1)
    update_text()

title_label = tk.Label(main_frame, text="", font=("Comfortaa", int(90 * scaling_factor), "bold"), bg="#1C1C1C",
                       fg="#00BFFF")
title_label.pack(pady=(int(30 * scaling_factor), int(10 * scaling_factor)))
typing_animation(title_label, "Cata-Detect\nKnow Your Eyes", delay=50)

# Create a description label
description_label = tk.Label(main_frame, text="Upload an image to detect cataracts.",
                             font=("Comfortaa", int(36 * scaling_factor)), bg="#1C1C1C", fg="#87CEEB")
description_label.pack(pady=int(30 * scaling_factor))

# Create a button to open an image file
open_button = CustomButton(main_frame, text="Open Image", command=open_file)
open_button.config(fg="#1E90FF")  # Set text color to blue
open_button.pack(pady=int(45 * scaling_factor))

# Create a label for loading animation
animation_label = tk.Label(main_frame, bg="#1C1C1C", font=("Comfortaa", int(36 * scaling_factor)), fg="#00BFFF")
animation_label.pack(pady=int(30 * scaling_factor))

# Create instructions label
instructions_label = tk.Label(main_frame,
                              text="Instructions:\n1. Click 'Open Image' to upload an image file.\n2. The system will analyze the image.\n3. Results will be shown in a pop-up window.",
                              font=("Comfortaa", int(30 * scaling_factor)), bg="#1C1C1C", fg="#FFFFFF", justify=tk.LEFT)
instructions_label.pack(pady=int(30 * scaling_factor))

# Load the model
model_path = 'model.pth'
model = load_model(model_path)

# Start the Tkinter event loop
root.mainloop()
