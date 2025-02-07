import gradio as gr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
from docx import Document
import cv2
import re
import pandas as pd


# Initialize PaddleOCR
ocr = PaddleOCR(
                use_angle_cls=True,
                lang='german'
                )


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


# preprocessing pipeline
def preprocess_image(image_path):
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



    

# Function to merge boxes in the same line
def merge_boxes(result, y_threshold=10):
    lines = []
    current_line = []
    
    # Sort boxes by y-coordinate
    sorted_boxes = sorted(result[0], key=lambda x: x[0][0][1])
    
    for box in sorted_boxes:
        if not current_line:
            current_line.append(box)
        else:
            # Compare y-coordinates to determine if they are in the same line
            last_box = current_line[-1]
            if abs(box[0][0][1] - last_box[0][0][1]) < y_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    if current_line:
        lines.append(current_line)
    
    # Merge boxes in each line
    merged_boxes = []
    for line in lines:
        x_coords = [point[0] for box in line for point in box[0]]
        y_coords = [point[1] for box in line for point in box[0]]
        merged_box = [
            [min(x_coords), min(y_coords)],
            [max(x_coords), min(y_coords)],
            [max(x_coords), max(y_coords)],
            [min(x_coords), max(y_coords)]
        ]
        merged_boxes.append(merged_box)
    
    return merged_boxes

# Perform text recognition on merged boxes
def extract_text_merged_boxes(merged_boxes, image):

    lines = []
    for box in merged_boxes:
        # Crop the region from the image
        x_min, y_min = int(box[0][0]), int(box[0][1])
        x_max, y_max = int(box[2][0]), int(box[2][1])
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # Perform OCR on the cropped image
        ocr_result = ocr.ocr(cropped_image, det=False, rec=True)
        for line in ocr_result:
            lines.append(line)

    cleaned_lines =  [line for line in lines if line is not None]
    extracted_text = [line[0][0] for line in cleaned_lines]
    return extracted_text

# ocr process pipeline
def ocr_process_img(image_path):
    
    image = cv2.imread(image_path)

    # Perform OCR to get detection boxes and text
    result = ocr.ocr(image)
    # merge boxes in each line
    merged_boxes = merge_boxes(result)
    # extract receipt data as text list
    extracted_text = extract_text_merged_boxes(merged_boxes, image)

    return extracted_text

# Function to extract products, prices and total
def extract_products_and_total(text_list):
    separated_data = []
    products = []
    residues = []
    total_price = None
    total_price_line = None
    
    # Define regex pattern for single product entries (product name + price + currency)
    product_pattern = re.compile(r'^(?P<name>[\w\s\.\-]+?)\s*(?P<price>\d*\.?\d+)\s*(B|C|Cn|Bn)$', re.UNICODE)

    # Define regex pattern for quantity-based product entries (quantity X name + unit price + total price + currency)
    quantity_pattern = re.compile(r'^(?P<quantity>\d*)\s*[Xx]\s*(?P<noise>[\w\s\.\-\:]+?)\s*(?P<unit_price>\d*\.?\d+)\s*(?P<price>\d*\.?\d+)\s*(B|C|Cn|Bn)$', re.UNICODE)
    
    # Define regex pattern to match total price (common keywords like "SUMME" or "EUR")
    total_price_pattern = re.compile(r'\b(EUR|SUMME|SUHHE|S[UO]MM?E|E[UO]R|SUH+E)\s*(EUR)?\s*(?P<total>\d+\.\d{2})$')
    
    # Add  products, quantities, or total price in a list
    for line in text_list:
       
        if product_pattern.match(line):
            products.append('_'+line)
            residues = []

        elif quantity_pattern.match(line):
            if residues:
                products.append(residues[-1]+'_'+line)
                residues = []

        elif total_price_pattern.match(line):
            total_price_line = line
            break

        else : 
            residues.append(line)
            if len(residues) == 1 and products:
                products[-1] = line+' '+products[-1]

    # Refined regex for parsing quantity-based products
    quantity_pattern = re.compile(r'^(?P<name>[\w\s\.\-]+?)_(?P<quantity>\d*)\s*[Xx]\s*(?P<noise>[\w\s\.\-\:]+?)\s*(?P<unit_price>\d*\.?\d{2})\s*(?P<price>\d*\.?\d{2})\s*(B|C|Cn|Bn)$', re.UNICODE)

    # Extract product details from products list
    for product in products:
        
        match_product_quantanty = re.match(quantity_pattern, product)
        match_product = re.match(product_pattern, product)

        if match_product_quantanty:
            product_name = match_product_quantanty.group(1).strip()
            quantity = match_product_quantanty.group(2)
            price = float(match_product_quantanty.group(5))
            separated_data.append({'Product_Name': product_name, 'Price': price, 'Quantity': quantity})

        elif match_product:
            product_name = match_product.group(1).strip()
            price = float(match_product.group(2))
            separated_data.append({'Product_Name': product_name, 'Price': price, 'Quantity': 1})

    # Extract total price if found
    if total_price_line :
        match_total = re.match(total_price_pattern, total_price_line)
        if match_total:
            total_price = match_total.group(3)
    # Append total price as a final entry
    separated_data.append({'Product_Name': "Total price", 'Price': total_price})
    
    return separated_data

def main_pipeline(image_path):

    preprocess_image_path = preprocess_image(image_path)

    extracted_text = ocr_process_img(preprocess_image_path)

    data = extract_products_and_total(extracted_text)

    
    df = pd.DataFrame(data)
    if df.empty:
        print("DataFrame is empty. Check extracted data.")

    # Clean Product Name
    df['Product_Name'] = df['Product_Name'].str.replace('_', ' ')
    # Save to Excel
    excel_path = "outputs/products.xlsx"
    df.to_excel(excel_path, index=False)

    return df,excel_path


# Create Gradio interface
iface = gr.Interface(
    fn=main_pipeline,
    inputs=gr.Image(type="filepath", label="Upload Receipt Image"),
    outputs=[
        gr.Dataframe(label="Extracted Text Structure"),
        gr.File(label="Download Excel File")
    ],
    title="Receipt OCR with PaddleOCR",
    description="Upload a receipt image to extract text and download it as a Text document."
)

# Launch the app
iface.launch()
