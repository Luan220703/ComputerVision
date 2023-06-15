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
image = cv2.imread(r"C:\Users\HOME\Downloads\234.jpg")

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



def apply_canny(image, threshold1, threshold2):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)

    return edges

# Load the image
image = cv2.imread(r"C:\Users\HOME\Downloads\Tuan cui 2.jpg")

# Apply Canny edge detection
threshold1 = 50  # Lower threshold
threshold2 = 150  # Upper threshold
edges = apply_canny(image, threshold1, threshold2)

# Display the original image and the edges
cv2.imshow("Original Image", image)
cv2.imshow("Canny Edges", edges)
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



