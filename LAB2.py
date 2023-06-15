import cv2 as cv
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import matplotlib.pyplot as plt

def color_balance(image, red_scale, green_scale, blue_scale):
    # Split the image into its color channels
    b, g, r = cv.split(image)

    # Scale the color channels using the provided parameters
    b = cv.multiply(b, blue_scale)
    g = cv.multiply(g, green_scale)
    r = cv.multiply(r, red_scale)

    # Clip the color values to the valid range (0-255)
    b = np.clip(b, 0, 255)
    g = np.clip(g, 0, 255)
    r = np.clip(r, 0, 255)

    # Merge the color channels back into a single image
    balanced_image = cv.merge((b, g, r))

    return balanced_image

def update_image(event):
    # Get the scaling factors from the sliders
    red_value = red_scale.get()
    green_value = green_scale.get()
    blue_value = blue_scale.get()

    # Perform color balance with the specified parameters
    balanced_image = color_balance(img, red_value, green_value, blue_value)

    # Display the balanced image
    cv.imshow('Color Balanced Image', balanced_image)

def perform_color_balance():
    # Load the input image
    image_path = filedialog.askopenfilename(initialdir='/', title='Select Image',
                                            filetypes=(('Image Files', '*.jpg *.jpeg *.png *.bmp'), ('All Files', '*.*')))
    if image_path:
        global img
        img = cv.imread(r'C:\Users\HOME\Downloads\Tuan cui 2.jpg')

        # Create a GUI window
        window = tk.Tk()
        window.title("Color Balance")
        window.geometry("400x250")

        # Create sliders for color balance adjustment
        global red_scale, green_scale, blue_scale
        red_scale = tk.Scale(window, from_=0, to=2, resolution=0.1, label="Red Scale", orient=tk.HORIZONTAL, command=update_image)
        red_scale.set(1.0)
        red_scale.pack(pady=10)

        green_scale = tk.Scale(window, from_=0, to=2, resolution=0.1, label="Green Scale", orient=tk.HORIZONTAL, command=update_image)
        green_scale.set(1.0)
        green_scale.pack(pady=10)

        blue_scale = tk.Scale(window, from_=0, to=2, resolution=0.1, label="Blue Scale", orient=tk.HORIZONTAL, command=update_image)
        blue_scale.set(1.0)
        blue_scale.pack(pady=10)
        cv.imshow('Original Image', img)
        window.mainloop()


def histogram_equalization(img):
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Perform histogram equalization
    equalized = cv.equalizeHist(gray)

    # Convert the equalized image back to BGR
    equalized_image = cv.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return equalized_image


def show_histogram(img, titles):
    # Calculate and plot the histograms of the images
    plt.figure(figsize=(12, 4))

    for i in range(len(img)):
        plt.subplot(1, len(img), i + 1)
        plt.hist(img[i].flatten(), bins=256, range=[0, 256], color='r')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()


def perform_histogram_equalization():
    # Load the input image

    img = cv.imread(r'C:\Users\HOME\Downloads\Tuan cui 2.jpg')
    # Load the image in grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Perform histogram equalization
    equalized_image = cv.equalizeHist(gray)

    # Display the original and equalized images
    cv.imshow('Original Image', gray)
    cv.imshow('Equalized Image', equalized_image)

    # Show the histograms
    show_histogram([gray, equalized_image], ['Original Image', 'Equalized Image'])


def salt_and_pepper_noise(img,bShow=False):
    """
    add noise salt and pepper noise into image
    :param img: input image
    :param bShow: whether to show the image or not
    :return: noised image
    """
    row, col = img.shape
    num_of_pixel = row*col
    noised_img = np.array(img)
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(int(num_of_pixel/15), int(num_of_pixel/14))
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        noised_img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(int(num_of_pixel/15), int(num_of_pixel/14))
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        noised_img[y_coord][x_coord] = 0
    if bShow:
        cv.imshow("src",img)
        cv.imshow("noise img",noised_img)
        cv.waitKey(0)
    return noised_img
def medianFilter(img,size=3,bShow=False):
    """Median filter( specilise for salt-pepper noise)
    :param img: input image
    :param size: kernel size
    :return: denoise image"""
    assert len(img.shape) == 2, "Input must be a gray image"
    rows,cols = img.shape
    new_img = np.zeros(img.shape,np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pixel_values = []
            for i in range(int(-size/2),int(size/2+0.5)):
                for j in range(int(-size/2),int(size/2+0.5)):
                    neighbor_row = row+i
                    neighbor_col = col+j
                    if neighbor_row>=0 and neighbor_col>=0 \
                            and neighbor_row < rows and neighbor_col < cols:
                        pixel_values.append(img[neighbor_row,neighbor_col])
            pixel_values = np.sort(np.array(pixel_values))
            med_idx = len(pixel_values)//2
            new_img[row,col] = pixel_values[med_idx]
    if bShow:
        cv.imshow("src", img)
        cv.imshow("median filter img", new_img)
        cv.waitKey(0)
    return new_img
def medianFilter_demo():
    """
    Demo for median filtering for salt-pepper noised image
    :return:
    """
    img = cv.imread(r'C:\Users\HOME\Downloads\Tuan cui 2.jpg', cv.IMREAD_GRAYSCALE)
    noised_img = salt_and_pepper_noise(img)
    medianFilter(noised_img, size=3, bShow=True)
def meanFilter(img,size=3,bShow=False):
    """
    Filtering image using average pixel values
    :param img: noise image
    :param size: size of kernel
    :param bShow: whether to show image or not
    :return: denoise image
    """
    assert len(img.shape)==2, "Input must be a gray image"
    rows,cols = img.shape
    new_img = np.zeros(img.shape,np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            sum = 0
            #print(pixel_values,int(-size/2),int(size/2+0.5))
            for i in range(int(-size/2),int(size/2+0.5)):
                for j in range(int(-size/2),int(size/2+0.5)):
                    neighbor_row = row+i
                    neighbor_col = col+j
                    if neighbor_row>=0 and neighbor_col>=0 \
                            and neighbor_row < rows and neighbor_col < cols:
                        sum += img[neighbor_row,neighbor_col]
            new_img[row,col] = sum//(size**2)
    if bShow:
        cv.imshow("src", img)
        cv.imshow("mean filter img", new_img)
        cv.waitKey(0)
    return new_img

def meanFilter_demo():
    """
    Demo for mean filtering
    :return:
    """
    img = cv.imread(r'C:\Users\HOME\Downloads\Tuan cui 2.jpg', cv.IMREAD_GRAYSCALE)
    noised_img = salt_and_pepper_noise(img)
    meanFilter(noised_img, size=3, bShow=True)
def gaussianBlurDemo():
    """
    Demo for gaussian blur
    :return:
    """
    img = cv.imread(r'C:\Users\HOME\Downloads\Tuan cui 2.jpg')

    cv.imshow("src", img)
    gauKernel = np.array([[ 1, 2,  1],
                            [ 2,  4, 2],
                            [ 1, 2,  1]])/16
    gauImg = cv.filter2D(img,ddepth=-1,kernel=gauKernel)
    cv.imshow("Gaussian blurS image",gauImg)
    cv.waitKey(0)
def main():
    img = cv.imread(r'C:\Users\HOME\Downloads\Tuan cui 2.jpg')
    perform_color_balance()
    perform_histogram_equalization()
    medianFilter_demo()
    gaussianBlurDemo()
    meanFilter_demo()


main()
