import cv2
import numpy as np
import matplotlib.pyplot as plt

def watershed_segmentation(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to remove noise and smooth the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Perform distance transform to obtain the distance map
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Convert the sure foreground to uint8 type
    sure_fg = np.uint8(sure_fg)

    # Find the unknown region
    unknown = cv2.subtract(opening, sure_fg)

    # Perform marker-based watershed segmentation
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    # Apply color mapping to visualize the segmented regions
    segmented = np.zeros_like(image)
    segmented[markers > 1] = [0, 0, 255]  # Set segmented regions to red color
    return segmented

def count_objects(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to remove noise and smooth the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Perform distance transform to obtain the distance map
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Convert the sure foreground to uint8 type
    sure_fg = np.uint8(sure_fg)

    # Find the unknown region
    unknown = cv2.subtract(opening, sure_fg)

    # Perform marker-based watershed segmentation
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    # Apply color mapping to visualize the segmented regions
    segmented = np.zeros_like(image)
    segmented[markers > 1] = [0, 0, 255]  # Set segmented regions to red color
    # Count the Object
    num_objects = len(np.unique(markers)) - 1
    return num_objects

# Path to the input image
image_path = r'C:\Users\HOME\Downloads\square.png'

# Perform watershed segmentation on the image
segmented_image = watershed_segmentation(image_path)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Apply watershed algorithm and count the objects
num_objects = count_objects(image_path)

print(f"Number of objects: {num_objects}")





def kmeans_segmentation(image, num_clusters):
    # Flatten the image
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape the labels to match the original image shape
    labels = labels.reshape(image.shape[:2])

    # Create segmented image using the cluster centers
    segmented_image = np.zeros_like(image)
    for i in range(num_clusters):
        segmented_image[labels == i] = centers[i]

    return segmented_image

# Load the image
image = cv2.imread(r'C:\Users\HOME\Downloads\1234567.jpg')

# Perform K-means segmentation with 5 clusters
num_clusters = 5
segmented_image = kmeans_segmentation(image, num_clusters)

# Display the result
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def meanshift_segmentation(image, spatial_radius, color_radius, min_density):
    # Convert the image to the required format for mean shift function
    shifted_image = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius, None, 2)

    # Convert the shifted image to grayscale
    gray = cv2.cvtColor(shifted_image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to obtain the segmentation mask
    _, mask = cv2.threshold(gray, min_density, 255, cv2.THRESH_BINARY)

    # Apply bitwise AND operation to obtain the segmented image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image

# Load the image
image = cv2.imread(r'C:\Users\HOME\Downloads\1234567.jpg')

# Set the parameters for mean shift segmentation
spatial_radius = 10
color_radius = 10
min_density = 100

# Perform Mean Shift segmentation
segmented_image = meanshift_segmentation(image, spatial_radius, color_radius, min_density)

# Display the result
cv2.imshow('Original Image',image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




