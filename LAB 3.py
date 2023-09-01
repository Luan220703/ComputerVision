import cv2
import numpy as np
import skimage
import skimage.feature as skif
from matplotlib import pyplot as plt


def harris_corner_detector(image, threshold=0.01, k=0.04):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the derivatives using the Sobel operator
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the products of derivatives at each pixel
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # Calculate the sums of the products of derivatives over a neighborhood
    window_size = 3
    sum_dx2 = cv2.boxFilter(dx2, -1, (window_size, window_size))
    sum_dy2 = cv2.boxFilter(dy2, -1, (window_size, window_size))
    sum_dxy = cv2.boxFilter(dxy, -1, (window_size, window_size))

    # Calculate the corner response function
    det_M = sum_dx2 * sum_dy2 - sum_dxy * sum_dxy
    trace_M = sum_dx2 + sum_dy2
    corner_response = det_M - k * trace_M * trace_M

    # Threshold the corner response function
    corner_mask = corner_response > threshold * corner_response.max()

    # Find the coordinates of the corners
    corners = np.argwhere(corner_mask)

    return corners

# Load the image
image = cv2.imread(r"C:\Users\HOME\Downloads\chess.jpg")

# Apply Harris corner detection
corners = harris_corner_detector(image)

# Draw circles at the detected corners
radius = 3
color = (0, 255, 0)  # Green
thickness = 2
for corner in corners:
    center = tuple(corner[::-1])
    cv2.circle(image, center, radius, color, thickness)

# Display the image with corners
cv2.imshow("Harris Corner Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def calculate_hog(image):
    # Convert the image to grayscale
    gray = skimage.color.rgb2gray(image)

    # Calculate the HOG features
    hog_features, hog_image = skif.hog(gray, visualize=True)

    return hog_features, hog_image

# Load the image
image = skimage.io.imread(r"C:\Users\HOME\Downloads\Tuan cui 2.jpg")

# Calculate the HOG features
hog_features, hog_image = calculate_hog(image)

# Display the HOG image and features
skimage.io.imshow(hog_image)
skimage.io.show()

print("HOG features shape:", hog_features.shape)
print("HOG features:", hog_features)



# Load the input image
image = cv2.imread(r"C:\Users\HOME\Downloads\Tuan cui 2.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Calculate gradients in the x and y directions using Sobel filters
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude and direction of gradients
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# Non-maximum suppression
edges = np.zeros_like(gradient_magnitude)
for i in range(1, gradient_magnitude.shape[0] - 1):
    for j in range(1, gradient_magnitude.shape[1] - 1):
        angle = gradient_direction[i, j]
        if (0 <= angle < np.pi/4) or (7*np.pi/4 <= angle < 2*np.pi):
            neighbor1 = gradient_magnitude[i, j + 1]
            neighbor2 = gradient_magnitude[i, j - 1]
        elif (np.pi/4 <= angle < 3*np.pi/4) or (5*np.pi/4 <= angle < 7*np.pi/4):
            neighbor1 = gradient_magnitude[i - 1, j]
            neighbor2 = gradient_magnitude[i + 1, j]
        elif (3*np.pi/4 <= angle < 5*np.pi/4) or (np.pi/4 <= angle < 3*np.pi/4):
            neighbor1 = gradient_magnitude[i - 1, j + 1]
            neighbor2 = gradient_magnitude[i + 1, j - 1]
        else:
            neighbor1 = gradient_magnitude[i - 1, j - 1]
            neighbor2 = gradient_magnitude[i + 1, j + 1]
        if gradient_magnitude[i, j] >= neighbor1 and gradient_magnitude[i, j] >= neighbor2:
            edges[i, j] = gradient_magnitude[i, j]

# Apply double thresholding and hysteresis to obtain final edges
low_threshold = 10
high_threshold = 100
edges = np.where(edges >= high_threshold, 255, np.where(edges >= low_threshold, 50, 0))

# Display the original image and the detected edges
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()



def detect_lines(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Apply Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10, maxLineGap=5)

    # Draw the detected lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return image

# Load the image
image = cv2.imread(r"C:\Users\HOME\Downloads\1234567.jpg")

# Detect lines in the image
result = detect_lines(image)

# Display the original image and the result
cv2.imshow("Line Detection Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()




