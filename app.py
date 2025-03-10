import gradio as gr

from pre_process import preprocess_receipt, preprocess_aadhar, preprocess_pan
from ocr_process import ocr_process_receipt, ocr_process_aadhar, ocr_process_pan
from post_process import post_process_receipt, post_process_aadhar, post_process_pan



def receipt_pipeline(image_path):

    preprocess_image_path = preprocess_receipt(image_path)

    extracted_text = ocr_process_receipt(preprocess_image_path)

    df,excel_path = post_process_receipt(extracted_text)

    return df,excel_path

def aadhar_pipeline(image_path):

    preprocess_image_path = preprocess_aadhar(image_path)

    extracted_text = ocr_process_aadhar(preprocess_image_path)

    df,excel_path = post_process_aadhar(extracted_text)

    return df,excel_path

def pan_pipeline(image_path):

    preprocess_image_path = preprocess_pan(image_path)

    extracted_text = ocr_process_pan(preprocess_image_path)

    df,excel_path = post_process_pan(extracted_text)

    return df,excel_path


def main_pipeline(document_type, image_path):
    if document_type == "Receipt":
        return receipt_pipeline(image_path)
    elif document_type == "Aadhar":
        return aadhar_pipeline(image_path)
    elif document_type == "PAN":
        return pan_pipeline(image_path)
    else:
        raise ValueError("Invalid document type selected.")

# Create Gradio interface
iface = gr.Interface(
    fn=main_pipeline,
    inputs=[
        gr.Dropdown(choices=["Receipt", "Aadhar", "PAN"], label="Select Document Type", value="Aadhar"),
        gr.Image(type="filepath", label="Upload Document Image"),
    ],
    outputs=[
        gr.Dataframe(label="Extracted Text Structure"),
        gr.File(label="Download Excel File")
    ],
    title="Document OCR with PaddleOCR",
    description="Upload a document image (Receipt, Aadhar, or PAN) to extract text and download it as an Excel file."
)

# Launch the app
iface.launch()
