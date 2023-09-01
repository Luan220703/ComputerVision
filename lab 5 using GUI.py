import PySimpleGUI as sg
import cv2
import numpy as np


# Function to align images using RANSAC algorithm
def align_images(source_img, target_img):
    # Convert images to grayscale
    gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and extract descriptors using ORB feature detector
    orb = cv2.ORB_create()
    keypoints_source, descriptors_source = orb.detectAndCompute(gray_source, None)
    keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, None)

    # Match keypoints using brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_source, descriptors_target)

    # Apply RANSAC to estimate the homography matrix
    ransac_reproj_thresh = 4.0  # RANSAC reprojection threshold
    src_pts = np.float32([keypoints_source[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)

    # Warp the source image to align with the target image
    aligned_img = cv2.warpPerspective(source_img, homography, (target_img.shape[1], target_img.shape[0]))
    print(np.shape(homography))

    return aligned_img


# GUI layout
layout = [
    [sg.Text("Source Image:"), sg.Input(key="-SOURCE-"), sg.FileBrowse()],
    [sg.Text("Target Image:"), sg.Input(key="-TARGET-"), sg.FileBrowse()],
    [sg.Button("Align"), sg.Button("Quit")]
]

# GUI window
window = sg.Window("Image Alignment", layout)

# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Quit":
        break
    if event == "Align":
        source_path = values["-SOURCE-"]
        target_path = values["-TARGET-"]

        # Load images
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)

        source_img = cv2.resize(source_img, (800, 600))
        target_img = cv2.resize(target_img, (800, 600))

        # Perform image alignment using RANSAC
        aligned_img = align_images(source_img, target_img)

        # Display aligned image
        cv2.imshow("Aligned Image", aligned_img)
        cv2.waitKey(0)

# Close GUI window
window.close()

