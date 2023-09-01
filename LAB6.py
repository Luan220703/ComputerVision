import cv2
import numpy as np

def stitch_images(images):
    # Initialize the SIFT feature detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors for the first image
    keypoints1, descriptors1 = sift.detectAndCompute(images[0], None)

    # Initialize the feature matcher
    matcher = cv2.BFMatcher()

    stitched_image = images[0]

    # Iterate over the remaining images
    for i in range(1, len(images)):
        # Find keypoints and descriptors for the next image
        keypoints2, descriptors2 = sift.detectAndCompute(images[i], None)

        # Match the keypoints between the current and previous images
        matches = matcher.match(descriptors1, descriptors2)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter out the top matches
        good_matches = matches[:50]

        # Extract the corresponding keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute the homography matrix using RANSAC
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp the next image to align with the previous image
        warped_image = cv2.warpPerspective(images[i], M, (stitched_image.shape[1], stitched_image.shape[0]))

        # Blend the warped image with the stitched image
        mask = np.where(warped_image != 0, 255, 0).astype(np.uint8)
        stitched_image = cv2.bitwise_and(stitched_image, cv2.bitwise_not(mask))
        stitched_image = cv2.add(stitched_image, warped_image)

        # Update the keypoints and descriptors for the next iteration
        keypoints1, descriptors1 = keypoints2, descriptors2

    return stitched_image

# Load the input images (scanned parts of the document)
image1 = cv2.imread(r'C:\Users\HOME\Downloads\foto1.jpg')
image2 = cv2.imread(r'C:\Users\HOME\Downloads\foto2.jpg')


# Resize the images (optional)
image1 = cv2.resize(image1, (800, 600))
image2 = cv2.resize(image2, (800, 600))


# Stitch the images
images = [image1, image2]
result = stitch_images(images)

# Display the stitched image

cv2.imshow('Stitched Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
