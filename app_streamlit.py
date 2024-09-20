import streamlit as st
from PIL import Image
import PyPDF2
import re
import json
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from datetime import datetime
from pyzbar.pyzbar import decode
import pytesseract
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import os
import tempfile
# from detectron2 import model_zoo
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.detectron2.data.datasets import register_coco_instances
from detectron2.detectron2.data import DatasetCatalog, MetadataCatalog
import pymupdf
import imageio
from DF_streamlit.cut_copy_detection import ForgeryDetection
from tempfile import NamedTemporaryFile
import folium
from streamlit_folium import folium_static
from st_aggrid import AgGrid
from geopy.geocoders import Nominatim
from src.ImageMetaData import ImageMetaData
from src.ocr import *
import time
from src.objectDetection import Detector
import io
import traceback
import fitz

# Set Tesseract path
#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\rosha\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to extract metadata from PDF
def extract_metadata_from_pdf(file):
    with file:
        parser = PDFParser(file)
        doc = PDFDocument(parser)
        return doc.info

def extract_year_month_day(date_strings):
    date_formats = [
        "%Y%m%d%H%M%S%z",  # Original format
        "%d.%m.%Y",  # Format in the text
        "%b %d, %Y",  # Example: Dec 28, 2022
        "%B %d, %Y",  # Example: December 28, 2022
        "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 format
        "%Y-%m-%d",  # yyyy-mm-dd
        "%d/%m/%Y",  # dd/mm/yyyy
        "%m/%d/%Y",  # mm/dd/yyyy

        
    ]
    formatted_dates = []
    for date_string in date_strings:
        date_string = date_string.get("CreationDate", b"").decode("utf-8")
        date_string = date_string.replace("D:", "").replace("'", "")
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_string, fmt)
                formatted_date = date_obj.strftime('%d.%m.%Y')
                formatted_dates.append(formatted_date)
                break
            except ValueError:
                continue
    return formatted_dates

# Function to extract date from PDF text
def extract_date_from_text(pdf_text):
    date_patterns = [
        r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b',  # dd.mm.yyyy
        r'\b(\d{4}-\d{2}-\d{2})\b',  # yyyy-mm-dd
        r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',  # Month dd, yyyy
        r'\b([A-Za-z]+ \d{1,2} , \d{4})\b',
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # mm/dd/yyyy or dd/mm/yyyy
        r'\b([A-Za-z]{3} \d{1,2} \d{4})\b'
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, pdf_text)
        if date_match:
            return date_match.group(1)
    return None

def convert_to_common_format(date_string):
    if date_string is None:
        return None
    
    date_formats = [
        "%d.%m.%Y",  # dd.mm.yyyy
        "%Y-%m-%d",  # yyyy-mm-dd
        "%b %d, %Y",  # Dec 28, 2022
        "%b %d %Y",
        "%B %d, %Y",  # December 28, 2022
        "%B %d , %Y",
        "%d/%m/%Y",  # dd/mm/yyyy
        "%m/%d/%Y",  # mm/dd/yyyy
    ]
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_string, fmt)
            return date_obj.strftime('%d.%m.%Y')
        except ValueError:
            continue
    return None

# Function to extract invoice info from QR data
def extract_invoice_info(qr_data):
    match_date = re.search(r"Invoice Date: (\d{2}-\d{2}-\d{4})", qr_data)
    match_total = re.search(r"Invoice Total: (\d+\.\d{2} USD)", qr_data)
    invoice_date = match_date.group(1) if match_date else None
    total_amount = match_total.group(1) if match_total else None
    return invoice_date, total_amount

# Function to extract QR code data from image
def extract_qr_code_data(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decoded_objects = decode(gray_image)
    qr_results = []
    for obj in decoded_objects:
        qr_data = obj.data.decode('utf-8')
        qr_results.append(extract_invoice_info(qr_data))
    return qr_results

# Function to extract text info from image using OCR
def extract_text_info(image):
    text = pytesseract.image_to_string(image)
    invoice_date_match = re.search(r"Invoice Date: (\d{2}-\d{2}-\d{4})", text)
    total_match = re.search(r"Total \$([\d.]+)", text)
    invoice_date = invoice_date_match.group(1) if invoice_date_match else None
    total_amount = total_match.group(1) if total_match else None
    return invoice_date, total_amount

# Function to analyze QR code in the image
def analyze_qr_code(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    qr_results = extract_qr_code_data(image_cv)
    return qr_results

def func_1(x,uploaded_file):
    detector = Detector(model_type=x)
    img = Image.open(uploaded_file).convert("RGB")
    #bytes_data = uploaded_file.read()
    #img = Image.open(io.BytesIO(bytes_data))
    if uploaded_file is not None:
        with NamedTemporaryFile(dir='.',suffix='.jpg',delete=False) as f:
            f.write(uploaded_file.getbuffer())
            out=detector.onImage(f.name)
              
            tout="INVOICE IS TAMPERED"
            if out==tout:
                st.markdown('<span style="color:red">**' + out + '**</span>', unsafe_allow_html=True)
                with st.expander("Proccesed Invoice"):
                    with open(uploaded_file.name,mode = "wb") as f: 
                        f.write(uploaded_file.getbuffer())
                        detector.onImage(f.name)
                        img_ = Image.open("result.jpg")
                        st.image(img_, caption='Proccesed Image.')

            else:
                st.markdown('<span style="color:green">**' + out + '**</span>', unsafe_allow_html=True)
        os.remove(f.name) 
    return out


def convert_pdf_to_images(pdf_file):
    images = []
    with pymupdf.open(pdf_file) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images
def func_3(x,file):
    ocr_summary = []
    path = file
    if path is not None:
        #path_name = imageio.imread(path)
        #with st.expander("**Uploaded Invoice**"):
            #st.write("**FileName :** ", path.name)
            #st.image(path)
        with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
            f.write(path.getbuffer())
             # Open the file using Pillow's Image module
            image = Image.open(path)

            # Convert the image to RGB (if it's not already)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save the image to the temporary file
            image.save(f.name, format='JPEG')
            data_ocr = OCR()
            result = data_ocr.qrReader(f.name)
            print("Result from QrReader:",result)
            print("******4")
            inv_data = data_ocr.ocrAPI(f.name)
            print("******5")
        # Check if QR code is detected
            if result is not None:
                st.success("QR Code Detected!")
                with st.spinner('Scanning QR...'):
                    time.sleep(1)
                    st.write("Available Data In QR Code:")
                    st.write(result)

                    # Call checkValues function and display result
                    data_to_check = result  # Replace with actual data to check
                    invoice_num = inv_data['invoice_no']
                    value_to_check = invoice_num  # Replace with actual value to check
                    result_match , check_result = data_ocr.checkValues(data_to_check, value_to_check)
                    invoice_dat = inv_data['invoice_date']
                    date_value_to_check = invoice_dat
                    dates_match , check_date = data_ocr.checkDates(result, date_value_to_check)
                    check_box = st.checkbox('**'+"Check Whether the QR Data and Invoice Data are same" +'**',key = f"{file.name}_ocr")
                    if check_box:
                        st.write("Invoice Number Matching :")
                        st.write(check_result)
                        st.write("Invoice Date Matching :")
                        st.write(check_date)
                    
                    #st.write("Result of checkValues function:")
                    #st.write(check_result)
                    #st.write("Result of checkDates function:")
                    #st.write(check_date)

                    # Call checkAmount method
                    inv_total_amt = inv_data['total_amt']
                    total_amt_value = inv_total_amt
                    inv_sum_amt = inv_data['sum_amt']
                    sum_amt_value = inv_sum_amt
                    amount_match , result_check_amount = data_ocr.checkAmount(result, total_amt_value, sum_amt_value)

                    # Display the result of checkAmount
                    st.info("**Total Amount spend Check :**")
                    st.text((result_check_amount))
                    if result_match and dates_match and amount_match:
                        ocr_summary.append({"Status": "No Tampering detected"})
                    else:
                        ocr_summary.append({"Status": "Tampering detected"})
                    
            else:
                st.warning("No QR Code found in the image.")
        os.remove(f.name)
        return ocr_summary


# Function to convert metadata to readable format
def convert_to_readable(metadata):
    readable_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8')
            
        if key.endswith('Date'):
            value = datetime.strptime(value, "D:%Y%m%d%H%M%S%z").strftime("%Y-%m-%d %H:%M:%S %Z")
            
        readable_metadata[key] = value
    return readable_metadata

def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredictor): Model to use.
            by the model.
        
    """
    # DatasetCatalog.clear()  # SN3 changed Commented
    # TRAIN_DATA_SET_NAME = f"data-train2"
    # TRAIN_DATA_SET_IMAGES_DIR_PATH = "fraud_detection-13/train"
    # TRAIN_DATA_SET_ANN_FILE_PATH = "fraud_detection-13/train/_annotations.coco.json"
    # with open(TRAIN_DATA_SET_ANN_FILE_PATH, "r") as f:
    #     coco_data = json.load(f)
    # class_names = [cat["name"] for cat in coco_data["categories"]]
    # print(class_names)
    # DatasetCatalog.clear()
    # register_coco_instances(
    #     name=TRAIN_DATA_SET_NAME,
    #     metadata={"thing_classes": class_names},
    #     json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    #     image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
    # )
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml"))
    yaml_path = "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    abs_yaml_path = os.path.abspath(os.path.join(os.getcwd(), yaml_path))
    cfg.merge_from_file(abs_yaml_path)  # SN3 changed
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # detectron2\detectron2\model_zoo\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml
    model_path = "model/model_0009999.pth"
    abs_model_path = os.path.abspath(os.path.join(os.getcwd(), model_path))
    cfg.MODEL.WEIGHTS = abs_model_path  # SN3 changed from "detectron2/model_0009999.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, predictor

def inference(predictor, img):
    return predictor(img)

def output_image(cfg, img, outputs, metadata):
    v = Visualizer(img[:, :, ::-1], metadata, scale=0.75)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img

def scrape_multi_font_text(file):
    results = []
    
    pdf = pymupdf.open(stream=file.getvalue(), filetype="pdf")
    
    for page_num, page in enumerate(pdf, start=1):
        page_dict = page.get_text("dict")
        blocks = page_dict["blocks"]
        
        for block in blocks:
            if "lines" in block.keys():
                spans = block["lines"]
                for span in spans:
                    data = span["spans"]
                    
                    fonts = set([line["font"] for line in data])
                    font_sizes = set([line["size"] for line in data])
                    
                    if len(fonts) > 1 or len(font_sizes) > 1:
                        x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                        text = " ".join([line["text"] for line in data])
                        
                        # Print the fonts and the text associated with them
                        for line in data:
                            print(f"Font: {line['font']}, Size: {line['size']}, Text: {line['text']}")
                        
                        results.append((text, x0, y0, x1, y1, page_num))
    
    pdf.close()
    return results

def draw_boxes(image, boxes):
    image_copy = np.copy(image)
    
    for box in boxes:
        _, x0, y0, x1, y1, _ = box
        cv2.rectangle(image_copy, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    
    return image_copy



# App Interface

def main():
    # Set page configuration
    st.set_page_config(
        page_title="DIGITRASE",
        page_icon="📊",
        layout="wide"
    )

    # Main title
    st.title("DIGITRASE")

    # Sidebar logo
    st.sidebar.image(r"DF_streamlit\Allianz.png", use_column_width=True)

    # Upload file
    files = st.sidebar.file_uploader("Upload File", type=["pdf", "jpg", "jpeg", "png"],accept_multiple_files = True)
    print(files)
    uploaded_files = files
    if files:
        st.sidebar.header("File Information")
        for file in files:
            st.sidebar.write(f"File Name: {file.name}")
            st.sidebar.text(f"Type: {file.type}")
            st.sidebar.text(f"Size: {round(len(file.getvalue()) / (1024 * 1024), 2)} MB")

        summary_data = []  
        #func_1("objectDetection",files)
        for file_no,file in enumerate(files):
            bytes_data = file.getvalue()
            bytes_stream = io.BytesIO(bytes_data)
            #st.write(file)
            #directory_name = os.path.dirname(file.name)
            #st.write(directory_name)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # Remove the file extension from the filename
            name, ext = os.path.splitext(file.name)
            # Combine the filename without extension and the timestamp
            file_id = f"{name}_{timestamp}"
            metadata_status = None
            qr_status = None
            detectron_status = None
            font_status = None

            pdf_metadata_status = None
            pdf_objectdetection_status = None
            pdf_qr_status = None
            pdf_detectron_status = None


            # Check file type
            if file.type == "application/pdf":
                    st.header(f"Filename: {file.name}")

                    # Convert PDF to images for QR code analysis and image segmentation
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(file.read())

                    # Convert PDF to images
                    poppler_path_1 = 'poppler-24.02.0/Library/bin'
                    abs_popler_path = os.path.abspath(os.path.join(os.getcwd(), poppler_path_1))     
                    pdf_images = convert_from_path(temp_path, 500, poppler_path=abs_popler_path)  # SN3 changed
                    images = convert_pdf_to_images(temp_path)
                    image_path = temp_path
                    # Remove the temporary file after conversion
                    os.remove(temp_path)
                    expand_checkbox = st.checkbox("Show uploaded image")
                    if expand_checkbox:
                        st.image(pdf_images, caption="Uploaded Image", width = 500)  # Expand the Image

                    # Metadata Analysis for PDF
                    pdf_text = extract_text_from_pdf(file)
                    pdf_metadata = extract_metadata_from_pdf(file)
                    metadata_date = extract_year_month_day(pdf_metadata)
                    date_from_text = extract_date_from_text(pdf_text)
                    # Convert extracted dates to common format for comparison
                    formatted_date_from_text = convert_to_common_format(date_from_text)
                    formatted_metadata_date = [convert_to_common_format(date) for date in metadata_date]

                    # Display PDF Metadata Analysis
                    pdf_metadata_results_summary = []
                    st.header("Metadata check")
                    st.text(pdf_text)
                    st.subheader("PDF Metadata:")
                    formatted_metadata = {key: (value.decode('utf-16') if isinstance(value, bytes) and (value.startswith(b'\xfe\xff') or value.startswith(b'\xff\xfe'))
                                                else value.decode('utf-8') if isinstance(value, bytes) else value if not isinstance(value, str) else value) for key, value in pdf_metadata[0].items()}
                    st.table(formatted_metadata)  #Meta data analysis Output 1
                    if formatted_metadata_date and formatted_date_from_text and formatted_metadata_date[0] == formatted_date_from_text:
                        st.success("Date matched in text and metadata!")
                        pdf_metadata_results_summary.append({"Status": "Date matched in text and metadata!"})  # SN3 added
                    else:
                        st.error("Date not matched between text and metadata.")
                        pdf_metadata_results_summary.append({"Status": "Date not matched between text and metadata."})  # SN3 added
                    pdf_metadata_status = any(result["Status"] == "Date not matched between text and metadata." for result in pdf_metadata_results_summary)  # SN3 added
                    
                    
                    st.header("Font Analysis")
                    
                    results = scrape_multi_font_text(bytes_stream)
                            

                    st.subheader("Scraped Results:")
                    if not results:
                        st.warning("No multi-font text found in the PDF.")
                    else:
                        # Draw bounding boxes on the first page
                        first_page = pymupdf.open(stream=bytes_stream, filetype="pdf").load_page(0)
                        img = first_page.get_pixmap()
                        img_array = np.frombuffer(img.samples, dtype=np.uint8).reshape((img.height, img.width, img.n)).copy()
                        img_with_boxes = draw_boxes(img_array, results)
                        st.image(img_with_boxes, caption='Text with Colored Bounding Boxes', use_column_width=True)   # Font Analysis Output 2

                        for result in results:
                            st.write(
                                f"Text: {result[0]}\n\n"
                                f"Coordinates: ({result[1]}, {result[2]}) - ({result[3]}, {result[4]}),\n\n Page: {result[5]}\n\n"
                                "---"
                            )

                    if results:
                        font_status == True
                    st.header("Pixel Manipulation Detection:")
                    st.header("Object Detection")
                    pdf_objectdetection_summary = []  # SN3 added
                    try:
                        for img in images:
                            if img is not None:
                                st.write("**FileName :** ",file.name)
                        
                                with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
                                    img.save(f, format='JPEG')
                                    detector = Detector(model_type="objectDetection")
                                    out = detector.onImage(f.name)

                                    tout = "INVOICE IS TAMPERED"
                                    if out == tout:
                                        st.markdown('<span style="color:red">**' + out + '**</span>', unsafe_allow_html=True)  # SN3 doubt on tout and out
                                        with st.expander("Processed Invoice"):
                                            print("****1")
                                            with open(file.name, mode="wb") as f:
                                                print("****2")
                                                f.write(file.getbuffer())
                                                print("****3")
                                                detector.onImage(f.name)
                                                img_ = Image.open("result.jpg")
                                                st.image(img_, caption='Processed Image.')    # Object Detection Output 3
                                        # pdf_objectdetection_summary.append({"Status": od_result})  # SN3 added doubt

                                    else:
                                        st.markdown('<span style="color:green">**' + out + '**</span>', unsafe_allow_html=True)  # SN3 doubt on tout and out
                                        # pdf_objectdetection_summary.append({"Status": od_result})  # SN3 added doubt
                                os.remove(f.name)
                    except Exception as e:
                        st.error(f"Error occurred during object detection analysis: {e}")

                    pdf_objectdetection_status = any(result["Status"] == "INVOICE IS TAMPERED" for result in pdf_objectdetection_summary)  # SN3 added

                    # QR Code Analysis for each PDF page
                    pdf_qr_summary=[]
                    # st.header("QR Code Analysis:")
                    # qr_results = analyze_qr_code(pdf_images)
                    
                    # if not qr_results:
                    #     st.warning("No QR code found in the image. Unable to perform QR code analysis.")
                    #     pdf_qr_summary.append({"Status": "No QR code found in the image. Unable to perform QR code analysis"})  # SN3 added
                    # else:
                    #     st.subheader("Results from QR Code Analysis:")
                    #     for qr_result in qr_results:
                    #         st.write(f"Invoice Date: {qr_result[0]}, Invoice Total: {qr_result[1]}")   # QR Code Ananlysis QR Date,etc Output 4
                                
                    #     text_results = extract_text_info(pdf_images)
                    #     st.subheader("Results from Text Extraction:")
                    #     st.write(f"Invoice Date: {text_results[0]}, Total: {text_results[1]} USD")  # QR Code Ananlysis Invoice Image Date,etc Output 4

                    #     # Compare results
                    #     if text_results[0] == qr_results[0][0] and text_results[1] == qr_results[0][1].split()[0]:
                    #         st.success("Results match!")
                    #         pdf_qr_summary.append({"Status": "Results match!"})  # SN3 added
                    #     else:
                    #         st.warning("Results do not match.")
                    #         pdf_qr_summary.append({"Status": "Results do not match."})  # SN3 added
                    
                    # pdf_qr_status = any(result["Status"] == "Results do not match." for result in pdf_qr_summary)  # SN3 added

                    # Detectron2 Analysis for each PDF page
                    pdf_detectron_results_summary=[]  # SN3 added
                    st.header("Instance Segmentation")
                    
                    for page_num, pdf_image in enumerate(pdf_images):
                        cfg, predictor = initialization()
                        metadata = MetadataCatalog.get("data-train2")
                        img_np = np.array(pdf_image)
                        outputs = inference(predictor, img_np)
                        num_instances = len(outputs['instances'])

                        if num_instances == 0:
                            st.success("No potential tampered regions found.")
                            pdf_detectron_results_summary.append({"Status": "No tampered regions"})  # SN3 added
                        else:
                            st.warning("The invoice appears to be tampered")
                            pdf_detectron_results_summary.append({"Status": "Tampered regions detected"})  # SN3 added
                            out_image = output_image(cfg, img_np, outputs, metadata)
                            st.image(out_image, caption=f'Processed Image - Page {page_num + 1}', use_column_width=True)  # Output 6
                    
                    pdf_detectron_status = any(result["Status"] == "Tampered regions detected" for result in pdf_detectron_results_summary)  # SN3 added

                    st.markdown(
                        """
                        <hr style="height:5px;border:none;color:#333;background-color:#333;" />
                        """,
                        unsafe_allow_html=True
                    )

                    # st.header("Copy Move Detection:") #SN3 commented
                    #try:

                    # print(type(file))  # tryexcluded 
                    # file_content = file.getbuffer()
                    # with tempfile.TemporaryDirectory() as temp_dir:
                    #     pdf_path = os.path.join(temp_dir, "uploaded_pdf.pdf")
                    #     with open(pdf_path, "wb") as f:
                    #         f.write(file_content)
                
                    

                    # #images = convert_from_path(pdf_path)
                    # for i, image in enumerate(images):
                    #     image_path = os.path.join(temp_dir, f"page_{i}.png")
                    #     image.save(image_path, "PNG")
                    #     st.image(image, caption=f"Page {i+1}")

                    #     # Apply forgery detection logic
                    #     file_image = cv2.imread(image_path)
                    #     detect = ForgeryDetection.Detect(file_image)
                    #     key_points, descriptors = detect.siftDetector()
                    #     forgery = detect.locateForgery(eps=20, min_sample=2)
                    #     st.info("Copy Paste Tampering Check in the above image")
                    #     st.image(forgery, caption="Processed Image")

                    #  SN3 added
                    pdf_tampering_modules = []
                    if pdf_metadata_status:
                        pdf_tampering_modules.append("PDF Metadata Analysis")
                    if pdf_qr_status:
                        pdf_tampering_modules.append("QR Code Analysis")
                    if pdf_detectron_status:
                        pdf_tampering_modules.append("Instance Segmentation Analysis")
                    if pdf_objectdetection_status:
                        pdf_tampering_modules.append("Object Detection Analysis")
                    if results:
                        pdf_tampering_modules.append("Font Analysis")
                    # if copymove_status:  #SN3 commented
                    #     tampering_modules.append("Copy-Move Detection")
                    # Joining the detected modules with HTML line break
                    tampering_modules_string = ", ".join(pdf_tampering_modules) if pdf_tampering_modules else "None"
                    summary_data.append({
                        "Sl. No.": file_no +1,
                        "File ID": file_id,
                        "Tampering detected modules": tampering_modules_string,
                        "Alert": "Yes" if any([pdf_metadata_status, not pdf_qr_status, pdf_detectron_status, pdf_objectdetection_status]) else "No"
                    })



            elif file.type in ["image/jpeg", "image/png", "image/jpg"]:
                st.header(f"File: {file.name}")
                expand_checkbox = st.checkbox("Show uploaded image",key = file.name)
                if expand_checkbox:
                    st.image(file, caption="Uploaded Image", width=500)

                # Metadata Analysis for Images
                st.header("Metadata check")

                try:
                    geolocator = Nominatim(user_agent="MyInvoiceForensicsApp/1.0")

                    # Load the image
                    image = Image.open(file).convert("RGB")

                    # Create a named temporary file
                    with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
                        # Save the uploaded file to the temporary file
                        file.seek(0)  # Ensure file pointer is at the beginning
                        f.write(file.read())

                        # Metadata analysis
                        meta_data = ImageMetaData(f.name)
                        exif_data = meta_data.get_exif_data()
                        latlng = meta_data.get_lat_lng()

                        if len(exif_data) != 0:
                            # Prepare metadata for display
                            metadata_dict = {"Fields": list(exif_data.keys()), "Values": list(exif_data.values())}
                            metadata_df = pd.DataFrame(metadata_dict)

                            # Display metadata using AgGrid
                            AgGrid(metadata_df,key=file.size)   # Output

                            # Display GPS coordinates and location if available
                            if all(latlng):
                                st.write("**The GPS Coordinates are:**", latlng)
                                Latitude, Longitude = str(latlng[0]), str(latlng[1])
                                location = geolocator.reverse(f"{Latitude},{Longitude}")
                                st.write("**The location of the incident:**", location)

                                # Display the location on a map
                                map_sby = folium.Map(location=[float(Latitude), float(Longitude)], zoom_start=15)
                                tooltip = location
                                folium.Marker([float(Latitude), float(Longitude)], popup=location, tooltip=tooltip).add_to(map_sby)
                                folium_static(map_sby)
                            else:
                                st.success("**GPS COORDINATES NOT AVAILABLE!!!**")
                        else:
                            st.success("**NO METADATA AVAILABLE!!!**")
                except Exception as e:
                    st.error(f"Error occurred during Metadata analysis: {e}")



                # OCR Analysis for Images
                st.header("QR code data match")
                ocr_summary = []
                try:
                    ocr_summary = func_3('ocr',file)
                    
                except Exception as e:
                    st.error(f"Error occurred during OCR analysis: {e}")
                # QR Code Analysis for images
                
                ocr_status = any(result["Status"] == "Tampering detected" for result in ocr_summary)

                # Display results for image analysis
                
                st.header("Pixel Manipulation Detection:")
                st.header("Instance Segmentation")
                detectron_results_summary = []
                try:
                    cfg, predictor = initialization()
                    metadata = MetadataCatalog.get("data-train2")
                    
                    img_np = np.array(image)
                    outputs = inference(predictor, img_np)
                    num_instances = len(outputs['instances'])
                    if num_instances == 0:
                        st.success("No potential tampered regions found.")
                        detectron_results_summary.append({"Status": "No tampered regions"})
                    else:
                        st.warning("The invoice appears to be tampered")
                        detectron_results_summary.append({"Status": "Tampered regions detected"})
                        out_image = output_image(cfg, img_np, outputs, metadata)
                        st.image(out_image, caption='Processed Image', use_column_width=True)  # Output
                
                except Exception as e:
                    st.error(f"Error occurred during instance segmentation analysis: {e}")

                detectron_status = any(result["Status"] == "Tampered regions detected" for result in detectron_results_summary)

                st.header("Object Detection")
                objectdetection_summary = []
                try:
                    od_result = func_1('objectDetection',file)
                    objectdetection_summary.append({"Status": od_result})
        
                except Exception as e:
                    st.error(f"Error occurred during object detection analysis: {e}")
                objectdetection_status = any(result["Status"] == "INVOICE IS TAMPERED" for result in objectdetection_summary)

                st.markdown(
                        """
                        <hr style="height:5px;border:none;color:#333;background-color:#333;" />
                        """,
                        unsafe_allow_html=True
                    )

                # # Copy Move Detection  #SN3 commented
                # st.header("Copy Move Detection:")
                # copymove_summary = []
                # try:
                #     with open(os.path.join("C:/Projects/DigiTrase/Document_Forensic_-obj_detection/DF_streamlit/cut_copy_detection/temp_img_dir/", file.name), "wb") as f:
                #             f.write(file.getbuffer())
                #     file_image_path = r'C:/Projects/DigiTrase/Document_Forensic_-obj_detection/DF_streamlit/cut_copy_detection/temp_img_dir/' + file.name
                #     file_name=imageio.imread(file_image_path)
                #     #st.write("**FileName :** ",file_name.name)
                #     #with st.expander("Show the Uploaded Invoice"):
                #         #st.image(file_name, caption='Uploaded Image.')
                        
                        
                #     eps = 20
                #     min_samples = 2

                #     # print('Detecting Forgery with parameter value as\neps:{}\nmin_samples:{}'.format(eps, min_samples))
                #     file_image_path = r'C:/Projects/DigiTrase/Document_Forensic_-obj_detection/DF_streamlit/cut_copy_detection/temp_img_dir/' + file.name
                #     file_image = cv2.imread(file_image_path)
                #     detect = ForgeryDetection.Detect(file_image)

                #     key_points, descriptors = detect.siftDetector()

                #     forgery, status = detect.locateForgery(eps, min_samples)
                    
                #     if status == True:
                #         copymove_summary.append({"Status": "TAMPERING FOUND"})
                #     else:
                #         copymove_summary.append({"Status": "NO TAMPERING FOUND"})
                #     st.info("Copy Paste Tampering check in the above image")
                #     st.image(forgery,caption = "Processedd Image")
                # except Exception as e:
                #     st.error(f"Error occurred during copy move detection: {e}")
                    

                # copymove_status = any(result["Status"] == "TAMPERING FOUND" for result in copymove_summary)
                
                tampering_modules = []
                if metadata_status:
                    tampering_modules.append("Metadata Analysis")
                if font_status:
                    tampering_modules.append("Font Analysis")
                if ocr_status:
                    tampering_modules.append("QR and OCR Analysis")
                #if qr_status:
                   #tampering_modules.append("QR Code Analysis")
                if detectron_status:
                    tampering_modules.append("Instance Segmentation Analysis")
                if objectdetection_status:
                    tampering_modules.append("Object Detection Analysis")
                # if copymove_status:  #SN3 commented
                #     tampering_modules.append("Copy-Move Detection")
                # Joining the detected modules with HTML line break
                tampering_modules_string = "<br>".join(tampering_modules) if tampering_modules else "None"
                summary_data.append({
                    "Sl. No.": file_no +1,
                    "File ID": file_id,
                    "Tampering detected modules": tampering_modules_string,
                    "Alert": "Yes" if any([metadata_status, font_status, ocr_status, detectron_status, objectdetection_status]) else "No"
                })
            
        st.markdown("""---""") 
        
        # CSS to style the floating button
        st.markdown("""
            <style>
            .floating-link {
                position: fixed;
                top: 50%;
                right: 20px;
                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                color: white !important;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 30px;
                text-align: center;
                text-decoration: none;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
                z-index: 9999;
            }
            .hidden {
                display: none;
            }
            </style>
            """, unsafe_allow_html=True)

        # JavaScript to hide the button after clicking
        st.markdown("""
            <script>
            function hideButton() {
                document.querySelector(".floating-link").classList.add("hidden");
            }
            </script>
            """, unsafe_allow_html=True)

        # Create the floating link to the summary report
        st.markdown('<a href="#summary-report" class="floating-link" style="color: white;" onclick="hideButton()">Go to Summary</a>', unsafe_allow_html=True)

        # Other Streamlit content here
        st.write("Some content here...")

        # Convert the data into a DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Convert DataFrame to Markdown table
        summary_table = summary_df.to_markdown(index=False)

        # Assign an ID to the summary report section
        st.markdown('<h2 id="summary-report">Summary Report:</h2>', unsafe_allow_html=True)

        # Display the Markdown table
        st.markdown(summary_table, unsafe_allow_html=True)
        
        #Add download button for CSV
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label = "Download summary as CSV",
            data = csv,
            file_name = 'summary_report.csv',
            mime='text/csv',
        )

if __name__ == '__main__':
		main()
