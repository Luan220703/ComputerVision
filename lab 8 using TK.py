import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import os


# Function to load images from a directory and convert them to grayscale
def load_images(directory):
    images = []
    labels = []
    label = 0

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img_gray)
            labels.append(label)
        label += 1

    return images, labels


# Function to preprocess the images and perform PCA
def preprocess(images):
    # Flatten the images into 1D arrays
    flattened_images = [img.flatten() for img in images]

    # Perform PCA
    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(flattened_images)

    # Project the images onto the eigenspace
    projected_images = pca.transform(flattened_images)

    return pca, projected_images


# Function to recognize faces in an input image
def recognize_faces(input_image, pca, knn_classifier):
    # Convert the input image to grayscale
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the input image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(input_image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform face recognition for each detected face
    for (x, y, w, h) in faces:
        face_roi = input_image_gray[y:y + h, x:x + w]

        # Resize the face region to match the training image size
        resized_face_roi = cv2.resize(face_roi, (img_width, img_height))

        # Flatten and project the resized face onto the eigenspace
        flattened_face = resized_face_roi.flatten()
        projected_face = pca.transform([flattened_face])

        # Predict the label of the face using the KNN classifier
        label = knn_classifier.predict(projected_face)

        # Draw a rectangle around the face and display the label
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(input_image, str(label[0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the input image with recognized faces
    cv2.imshow("Recognized Faces", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to handle the mouse click event on the image
def handle_click(event):
    global input_image_path, pca, knn_classifier

    # Open the file dialog to choose an image file
    file_path = filedialog.askopenfilename()

    if file_path:
        # Load the selected image
        input_image = cv2.imread(file_path)

        # Perform face recognition on the selected image
        recognize_faces(input_image, pca, knn_classifier)


# Create the GUI window
window = tk.Tk()
window.title("Eigenfaces Face Recognition")
window.geometry("400x300")

# Create a button to choose the input image
button = tk.Button(window, text="Select Image", command=handle_click)
button.pack(pady=10)
train_dir = r'C:\Users\HOME\webcam_face_recognition-master\faces'

# Load and preprocess the training images
train_images, train_labels = load_images(train_dir)
pca, projected_images = preprocess(train_images)

# Train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(projected_images, train_labels)

# Set the dimensions for resizing the face region
img_width = 100
img_height = 100

# Start the GUI event loop
window.mainloop()



