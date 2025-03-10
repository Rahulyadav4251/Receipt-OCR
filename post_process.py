import pandas as pd
import re

# Function to extract products, prices and total from receipts

def extract_receipt_details(text_list):
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

# Phrase to exclude
BLACKLIST_aadhar = {"government of india", "hr", "art"}

# Define the regex patterns
aadhar_name_pattern = re.compile(r'^(?P<name>[A-Za-z]+(?:\s[A-Za-z]+)*)$', re.UNICODE) 
aadhar_dob_pattern = re.compile(r'(?P<dob>\d{2}/\d{2}/\d{4})')
aadhar_gender_pattern = re.compile(r'\b(?P<gender>Male|Female)\b', re.IGNORECASE)
aadhar_pattern = re.compile(r'\b(?P<aadhar>\d{4}\s?\d{4}\s?\d{4})\b')

def extract_aadhar_details(ocr_text_list):
    '''Function to extract Name, DOB, Gender, and Aadhar Number from OCR text.'''


    name = None
    dob = None
    gender = None
    aadhar = None

    for text in ocr_text_list:

        # Skip blacklisted phrases
        if text.lower() in BLACKLIST_aadhar:
            continue

        # Check for name 
        if aadhar_name_pattern.match(text) and len(text.split()) >= 2:
            name = text.strip()

        # Check for DOB
        elif aadhar_dob_pattern.search(text):
            dob_match = aadhar_dob_pattern.search(text)
            dob = dob_match.group('dob')

        # Check for gender
        elif aadhar_gender_pattern.search(text):
            gender_match = aadhar_gender_pattern.search(text)
            gender = gender_match.group('gender')

        # Check for Aadhar number
        elif aadhar_pattern.search(text):
            aadhar_match = aadhar_pattern.search(text)
            aadhar = aadhar_match.group('aadhar').replace(" ", "")  # Remove spaces

    return {"Name": name, "DOB": dob, "Gender": gender, "Aadhar": aadhar}


# Phase to exclude
BLACKLIST_pan = {
    'permanent account number', 'hrror', 'income tax department', 'govt.of india', 'e-permanent account number card', 
    '/name', '/fathers name', '/date of birth', '/signature', 'e-permanent account number card', 
    'signature', 'date of birth', 'fathers name', 'name', 'government of india', 
    'govt. of india', 'income tax department', 'permanent account number card', 'hr', 'art'
}
# Define the regex patterns
pan_name_pattern = re.compile(r'^(?P<name>[A-Za-z]+(?:\s[A-Za-z]+)*)$', re.UNICODE) 
pan_dob_pattern = re.compile(r'(?P<dob>\d{2}/\d{2}/\d{4})')

pan_pattern = re.compile(r'(?P<pan>^[A-Z]{5}\d{4}[A-Z]$)')

def extract_pan_details(ocr_text_list):
    '''Function to extract Name, DOB, Gender, and Aadhar Number from OCR text.'''

    name_list = []
    dob = None
    pan = None

    for text in ocr_text_list:

        # Skip blacklisted phrases
        if text.lower().strip() in BLACKLIST_pan:
            continue

        # Check for name 
        if pan_name_pattern.match(text.strip()):
            name_list.append(text.strip())

        # Check for DOB
        elif pan_dob_pattern.search(text):
            dob_match = pan_dob_pattern.search(text)
            dob = dob_match.group('dob')


        # Check for pan number
        elif pan_pattern.search(text):
            pan_match = pan_pattern.search(text)
            pan = pan_match.group('pan').replace(" ", "")  # Remove spaces
    
    if len(name_list)>=2 :
        name = name_list[-2]
        parent_name = name_list[-1]
    else:
        name = None
        parent_name =None

    return {"Name": name, "DOB": dob, "Parent": parent_name, "Pan": pan}



# post processing pipeline for receipt
def post_process_receipt(extracted_text):

    data = extract_receipt_details(extracted_text)

    df = pd.DataFrame(data)
    if df.empty:
        print("DataFrame is empty. Check extracted data.")

    # Clean Product Name
    df['Product_Name'] = df['Product_Name'].str.replace('_', ' ')
    # Save to Excel
    excel_path = "outputs/receipt.xlsx"
    df.to_excel(excel_path, index=False)

    return df,excel_path


# post processing pipeline for aadhar
def post_process_aadhar(extracted_text):

    data = extract_aadhar_details(extracted_text)

    columns = ["name", "dob", "gender", "aadhar"]
    aadhar_df = pd.DataFrame(columns=columns)

    # Create a DataFrame for the new row
    new_row = pd.DataFrame([{
        "name": data["Name"],
        "dob": data["DOB"],
        "gender": data["Gender"],
        "aadhar": data["Aadhar"]
    }])

    aadhar_df = pd.concat([aadhar_df, new_row], ignore_index=True)

    # Save to Excel
    excel_path = "outputs/aadhar.xlsx"
    aadhar_df.to_excel(excel_path, index=False)

    return aadhar_df,excel_path


# post processing pipeline for pan
def post_process_pan(extracted_text):

    data = extract_pan_details(extracted_text)

    columns = ["name", "dob", "parent", "pan_id"]
    pan_df = pd.DataFrame(columns=columns)


    # Create a DataFrame for the new row
    new_row = pd.DataFrame([{
        "name": data["Name"],
        "dob": data["DOB"],
        "parent": data["Parent"],
        "pan_id": data["Pan"]
    }])

    pan_df = pd.concat([pan_df, new_row], ignore_index=True)

    # Save to Excel
    excel_path = "outputs/pan.xlsx"
    pan_df.to_excel(excel_path, index=False)

    return pan_df,excel_path