import cv2
import numpy as np


def biggest_contour(contours):
    """
    Find the biggest contour with four sides.

    Parameters:
    contours (list): List of detected contours.

    Returns:
    np.array: The largest quadrilateral contour found.
    """
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:  # Ignore small contours
            peri = cv2.arcLength(i, True)  # Calculate perimeter
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)  # Approximate contour shape
            if area > max_area and len(approx) == 4:  # Check if it has 4 sides
                biggest = approx
                max_area = area
    return biggest

def correct_orientation(image_path):
    """
    Correct the orientation of the image using contour.

    Parameters:
    image_path (str): Path of input image.

    Returns:
    np.array: Corrected image as a NumPy array.
    """
    # Load the image (OpenCV reads in BGR format)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error loading image file.")

    img_original = img.copy()

    # Convert image to grayscale and reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)

    # Detect edges using Canny edge detection
    edged = cv2.Canny(gray, 10, 20)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep the top 10 largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Find the biggest contour that forms a quadrilateral
    biggest = biggest_contour(contours)

    # If no valid contour found, return original image
    if biggest.size == 0:
        print("No valid contour detected. Returning original image.")
        return img_original

    # Reshape contour points for transformation
    points = biggest.reshape(4, 2)

    # Order points correctly
    input_points = np.zeros((4, 2), dtype="float32")
    # Order the points: top-left, top-right, bottom-left, bottom-right
    points_sum = points.sum(axis=1)  # Sum of (x, y) coordinates
    input_points[0] = points[np.argmin(points_sum)]  # Top-left
    input_points[3] = points[np.argmax(points_sum)]  # Bottom-right

    points_diff = np.diff(points, axis=1)  # Difference (x - y)
    input_points[1] = points[np.argmin(points_diff)]  # Top-right
    input_points[2] = points[np.argmax(points_diff)]  # Bottom-left

    # Extract ordered points
    (top_left, top_right, bottom_right, bottom_left) = input_points

    # Compute dimensions for transformation
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    max_width = max(int(bottom_width), int(top_width))
    max_height = max(int(left_height), int(right_height))
    

    # Define destination points
    converted_points = np.float32([
        [0, 0], [max_width, 0], [0, max_height], [max_width, max_height]
    ])

    # Compute transformation matrix
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)

    # Apply perspective warp
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

   

    return img_output  # Returns a NumPy array 

def crop_image(image):
    """
    Crop out the Product , price, Total

    parameters:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Cropped image.
    """

    height, width = image.shape[:2]
    start_y = int(height * 0.12)  # Start from 10% from the top
    end_y = int(height * 0.55)    # End at 55% from the top
    # Crop the image
    cropped_image = image[start_y:end_y, 0:width]
    
    return cropped_image

def enhance_contrast_clahe(image):
    """
    Enhance the contrast of the image using CLAHE.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Contrast-enhanced image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced



# preprocessing pipeline for receipt
def preprocess_receipt(image_path):
    """
    Preprocess the image: correct orientation, cropping and enhance contrast.

    Parameters:
    image_path (str): Path to input image.

    Returns:
    str: Path to preprocessed image.
    """
   
    image = correct_orientation(image_path)

    cropped_image = crop_image(image)

    enhanced_image = enhance_contrast_clahe(cropped_image)

    # Save the preprocessed image 
    preprocessed_image_path = "outputs/preprocessed_image.png"
    cv2.imwrite(preprocessed_image_path, enhanced_image)

    return preprocessed_image_path 


# preprocessing pipeline for aadhar
def preprocess_aadhar(image_path):
    """
    Preprocess the image: correct orientation, cropping and enhance contrast.

    Parameters:
    image_path (str): Path to input image.

    Returns:
    str: Path to preprocessed image.
    """
    preprocessed_image_path = image_path


    return preprocessed_image_path 

# preprocessing pipeline for pan
def preprocess_pan(image_path):
    """
    Preprocess the image: correct orientation, cropping and enhance contrast.

    Parameters:
    image_path (str): Path to input image.

    Returns:
    str: Path to preprocessed image.
    """
    preprocessed_image_path = image_path
  

 

    return preprocessed_image_path 