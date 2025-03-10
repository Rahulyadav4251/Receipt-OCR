import cv2
from paddleocr import PaddleOCR, draw_ocr

# Initialize PaddleOCR english
ocr_en = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    )

# Initialize PaddleOCR german
ocr_ger = PaddleOCR(
    use_angle_cls=True,
    lang='german',
    )



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
def extract_text_merged_boxes(merged_boxes, image, ocr_model=ocr_en):

    lines = []
    for box in merged_boxes:
        # Crop the region from the image
        x_min, y_min = int(box[0][0]), int(box[0][1])
        x_max, y_max = int(box[2][0]), int(box[2][1])
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # Perform OCR on the cropped image
        ocr_result = ocr_model.ocr(cropped_image, det=False, rec=True)
        for line in ocr_result:
            lines.append(line)

    cleaned_lines =  [line for line in lines if line is not None]
    extracted_text = [line[0][0] for line in cleaned_lines]
    return extracted_text


# ocr process pipeline
def ocr_process_receipt(image_path, ocr_model = ocr_ger):
    
    image = cv2.imread(image_path)

    # Perform OCR to get detection boxes and text
    result = ocr_model.ocr(image)
    # merge boxes in each line
    merged_boxes = merge_boxes(result)
    # extract receipt data as text list
    extracted_text = extract_text_merged_boxes(merged_boxes, image, ocr_model)

    return extracted_text


def ocr_process_aadhar(image_path, ocr_model = ocr_ger):
    
    image = cv2.imread(image_path)

    results = ocr_model.ocr(image)
    extracted_text = [line[1][0] for line in results[0]]


    return extracted_text

def ocr_process_pan(image_path, ocr_model = ocr_ger):
    
    image = cv2.imread(image_path)

    results = ocr_model.ocr(image)

    results = ocr_model.ocr(image_path)
    extracted_text = [line[1][0] for line in results[0]]

    return extracted_text