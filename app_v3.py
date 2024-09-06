import os
from pdf2image import convert_from_path
from PIL import Image
from src.pdfMetadata import PDFExtractor
from src.ImageMetaData import ImageMetaData
from src.fontExtraction import FontExtractor
from src.objectDetection import Detector
from src.instanceSegmentation import Segmentor
from src.copymove import Detect
from src.ocr import OCR
import pymupdf
import numpy as np
import cv2
from tempfile import NamedTemporaryFile
import time
import tempfile
import json
from geopy.geocoders import Nominatim
from datetime import datetime

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Digitrase starting...")
    print("**********")

    # List files in the 'uploads' folder
    
    files = os.listdir(UPLOAD_DIR)
    # Filter files based on supported types
    supported_types = [".pdf", ".jpg", ".jpeg", ".png"]
    uploaded_files = [file for file in files if os.path.splitext(file)[1].lower() in supported_types]
    if uploaded_files:
        # Process the only file in the list (for multiple files loop can be given here)
        file = uploaded_files[0]
        file_path = os.path.join(UPLOAD_DIR, file)
        with open(file_path, 'rb') as file:
            file_name , file_extension = os.path.splitext(file_path)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_id = f"{file_name}_{timestamp}"
            print(file_name, file_extension)
            if file_extension == ".pdf":
                pdf_metadata_status = None
                pdf_ocr_status = None
                pdf_objectdetection_status = None
                pdf_instanceseg_status = None
                pdf_fontanalysis_status = None
                pdf_copymove_summary = None
                #print("file is pdf")
                poppler_path_1 = r'poppler-24.02.0/Library/bin'
                abs_popler_path = os.path.abspath(os.path.join(os.getcwd(), poppler_path_1))     
                pdf_images = convert_from_path(file_path, 500, poppler_path=abs_popler_path)
                #print("Pdf Converted to image")

                #for i, image in enumerate(pdf_images):
                    # Display the image
                    # image.show()
            

                print("PDF Metadata Analysis:")
                extractor = PDFExtractor()
                with open(file_path, "rb") as file:
                    pdf_text = extractor.extract_text_from_pdf(file)
                    pdf_metadata = extractor.extract_metadata_from_pdf(file)
                metadata_date = extractor.extract_year_month_day(pdf_metadata)
                date_from_text = extractor.extract_date_from_text(pdf_text)
                print("PDF Text:\n",pdf_text)

                # Convert extracted dates to common format for comparison
                formatted_date_from_text = extractor.convert_to_common_format(date_from_text)
                formatted_metadata_date = [extractor.convert_to_common_format(date) for date in metadata_date]
                
                formatted_metadata = {key: (value.decode('utf-16') if isinstance(value, bytes) and (value.startswith(b'\xfe\xff') or value.startswith(b'\xff\xfe'))
                                        else value.decode('utf-8') if isinstance(value, bytes) else value if not isinstance(value, str) else value) for key, value in pdf_metadata[0].items()}
                print("PDF Metadata:\n",formatted_metadata)  #Meta data analysis Output 1

                pdf_metadata_results_summary = []

                if formatted_metadata_date and formatted_date_from_text and formatted_metadata_date[0] == formatted_date_from_text:
                    pdf_metadata_results_summary.append({"Status": "Date matched in text and metadata!"})  
                else:
                    pdf_metadata_results_summary.append({"Status": "Date not matched between text and metadata."})  

                pdf_metadata_status = any(result["Status"] == "Date not matched between text and metadata." for result in pdf_metadata_results_summary)  # SN3 added
                print(pdf_metadata_results_summary[0]["Status"])

                print("QR code and OCR Analysis:")
                pdf_ocr_summary = []
                if file_path is not None:
                    for page_num, pdf_image in enumerate(pdf_images):
                    #path_name = imageio.imread(path)
                    #with st.expander("**Uploaded Invoice**"):
                        #st.write("**FileName :** ", path.name)
                        #st.image(path)
                        with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:

                            # Convert the image to RGB (if it's not already)
                            if pdf_image.mode != 'RGB':
                                pdf_image = pdf_image.convert('RGB')

                            # Save the image to the temporary file
                            pdf_image.save(f.name, format='JPEG')
                            data_ocr = OCR()
                            result = data_ocr.qrReader(f.name)
                            print("******4")
                            inv_data = data_ocr.ocrAPI(f.name)
                            print("******5")
                        # Check if QR code is detected
                            if result is not None:
                                print("QR Code Detected!")
                                print('Scanning QR...')
                                time.sleep(1)
                                print("Available Data In QR Code:\n",result)

                                # Call checkValues function and display result
                                data_to_check = result  # Replace with actual data to check
                                invoice_num = inv_data['invoice_no']
                                value_to_check = invoice_num  # Replace with actual value to check
                                result_match , check_result = data_ocr.checkValues(data_to_check, value_to_check)
                                invoice_dat = inv_data['invoice_date']
                                date_value_to_check = invoice_dat
                                dates_match , check_date = data_ocr.checkDates(result, date_value_to_check)
                                print('**'+"Checking Whether the QR Data and Invoice Data are same..." +'**')
                                
                                print("Invoice Number Matching :\n",check_result)
                                print("Invoice Date Matching :\n",check_date)
                    
                                    
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
                                print("**Total Amount spend Check :**")
                                print(result_check_amount)
                                if result_match and dates_match and amount_match:
                                    pdf_ocr_summary.append({"Status": "No Tampering detected"})
                                else:
                                    pdf_ocr_summary.append({"Status": "Tampering detected"})
                                    
                            else:
                                print("No QR Code found in the image.")
                        os.remove(f.name)
                        pdf_ocr_status = any(result["Status"] == "Tampering detected" for result in pdf_ocr_summary)

                
                print("Detectron2 Object Detection:")
                pdf_objectdetection_summary = []  # SN3 added
                try:
                    for image in pdf_images:
                        if image is not None:
                            with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
                                image.save(f, format='JPEG')
                                detector = Detector(model_type="objectDetection")
                                out = detector.onImage(f.name)

                                tout = "INVOICE IS TAMPERED"
                                if out == tout:
                                    print(out)
                                    print("Processed Invoice")
                                    print("****1")
                                    with open(file.name, mode="wb") as f:
                                        print("****2")
                                        f.write(file.getbuffer())
                                        print("****3")
                                        detector.onImage(f.name)
                                        img_ = Image.open("result.jpg")
                                        od_file_path = os.path.join(RESULTS_DIR, "Object_Detection_result.jpg")
                                        cv2.imwrite(od_file_path, img_)

                                else:
                                    print(out)  
                            os.remove(f.name)
                            pdf_objectdetection_summary.append({"Status": out})
                            pdf_objectdetection_status = any(result["Status"] == tout for result in pdf_objectdetection_summary)
                except Exception as e:
                    print(e)

                print("Instance Segmentation:\n")
                pdf_instanceseg_results_summary = []
                for page_num, pdf_image in enumerate(pdf_images):
                    segmentor = Segmentor()
                    img_np = np.array(pdf_image)
                    outputs = segmentor.inference(img_np)
                    num_instances = len(outputs['instances'])

                    if num_instances == 0:
                        print("No potential tampered regions found.")
                        pdf_instanceseg_results_summary.append({"Status": "No tampered regions"})  # SN3 added
                    else:
                        print("The invoice appears to be tampered")
                        pdf_instanceseg_results_summary.append({"Status": "Tampered regions detected"})  # SN3 added
                        out_image = segmentor.output_image(img_np, outputs)
                        print("Processed")
                        is_file_path = os.path.join(RESULTS_DIR, "Instance Segmentation result.jpg")
                        cv2.imwrite(is_file_path, out_image)
                        #st.image(out_image, caption=f'Processed Image - Page {page_num + 1}', use_column_width=True)  # Output 6
                    
                    pdf_instanceseg_status = any(result["Status"] == "Tampered regions detected" for result in pdf_instanceseg_results_summary)


                print("Copy-Move detection")
                pdf_copymove_summary = []
                for i, image in enumerate(pdf_images):
                    with NamedTemporaryFile(dir='.', suffix=f'_{i}.jpg', delete=False) as f:
                        # Convert the image to RGB (if it's not already)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')

                        # Save the image to the temporary file
                        image.save(f.name, format='JPEG')

                        try:
                            file_image_path = f.name
                            file_image = cv2.imread(file_image_path)
                            detect = Detect(file_image)

                            key_points, descriptors = detect.siftDetector()
                            forgery, status = detect.locateForgery(eps=20, min_sample=2)
                            
                            if status:
                                pdf_copymove_summary.append({"Status": "TAMPERING FOUND"})
                            else:
                                pdf_copymove_summary.append({"Status": "NO TAMPERING FOUND"})
                            
                            # Print or log the processed image if needed
                            # Here we simply print the status and save the image if tampering is found
                            print(status)
                            if status:
                                cv2.imwrite(f'forgery_detected_{i}.jpg', forgery)

                        except Exception as e:
                            print(f"Error occurred during copy move detection: {e}")
                    #copy_move status logic to be added
                    os.remove(f.name)

                print("Font Analysis:")
                pdf_fontanalysis_summary =[]
                scrapper = FontExtractor()
                print("scrapping fonts...")
                results = scrapper.scrape_multi_font_text(file_path)

                print("Scraped Results:")
                if not results:
                    print("No multi-font text found in the PDF.")
                else:
                    # Draw bounding boxes on the first page
                    first_page = pymupdf.open(file_path).load_page(0)
                    img = first_page.get_pixmap()
                    img_array = np.frombuffer(img.samples, dtype=np.uint8).reshape((img.height, img.width, img.n)).copy()
                    img_with_boxes = scrapper.draw_boxes(img_array, results)
                    # saving the result
                    fa_file_path = os.path.join(RESULTS_DIR, "Font Analysis result.jpg")
                    cv2.imwrite(fa_file_path, img_with_boxes)
                    for result in results:
                        print(
                            f"Text: {result[0]}\n\n"
                            f"Coordinates: ({result[1]}, {result[2]}) - ({result[3]}, {result[4]}),\n\n Page: {result[5]}\n\n"
                            "---"
                        )
                print(len(results))
                if len(results)>2:
                    pdf_fontanalysis_summary.append({"Status": "TAMPERING FOUND"})
                else:
                    pdf_fontanalysis_summary.append({"Status": "NO TAMPERING FOUND"})

                pdf_fontanalysis_status = any(result["Status"] == "TAMPERING FOUND" for result in pdf_fontanalysis_summary)
                
                summary_data = []
                tampering_modules = []
                if pdf_metadata_status:
                    tampering_modules.append("Metadata Analysis")
                if pdf_ocr_status:
                    tampering_modules.append("QR and OCR Analysis")
                if pdf_objectdetection_status:
                    tampering_modules.append("Object Detection Analysis")
                if pdf_instanceseg_status:
                    tampering_modules.append("Instance Segmentation Analysis")
                # if copymove_status:  # need to create logic to flag
                #     tampering_modules.append("Copy-Move Detection")
                if pdf_fontanalysis_status:
                    tampering_modules.append("Font Analysis")
                
                #if qr_status:
                    #tampering_modules.append("QR Code Analysis")
                
                # Joining the detected modules with HTML line break
                tampering_modules_string = ",".join(tampering_modules) if tampering_modules else "None"
                summary_data.append({
                    "Sl. No.": 1,
                    "File ID": file_id,
                    "Tampering detected modules": tampering_modules_string,
                    "Alert": "Yes" if any([pdf_metadata_status, pdf_ocr_status, pdf_objectdetection_status, pdf_instanceseg_status, pdf_fontanalysis_status]) else "No"
                })

                print(summary_data)

                json_file_path = os.path.join(RESULTS_DIR, 'summary_data.json')
                with open(json_file_path, 'w') as json_file:
                    json.dump(summary_data, json_file, indent=4)


            elif file_extension in [".jpg", ".jpeg", ".png"]:
                #print("file is image")

                img_metadata_status = None
                img_ocr_status = None
                img_objectdetection_status = None
                img_instanceseg_status = None
                img_copymove_summary = None

                image = Image.open(file_path).convert("RGB")

                print("Metadata Analysis")

                try:
                    geolocator = Nominatim(user_agent="MyInvoiceForensicsApp/1.0")

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
                            print(metadata_dict)
                            #metadata_df = pd.DataFrame(metadata_dict)

                            # Display metadata using AgGrid
                            #AgGrid(metadata_df,key=file.size)   # Output

                            # Display GPS coordinates and location if available
                            if all(latlng):
                                print("**The GPS Coordinates are:**", latlng)
                                Latitude, Longitude = str(latlng[0]), str(latlng[1])
                                location = geolocator.reverse(f"{Latitude},{Longitude}")
                                print("**The location of the incident:**", location)

                                # Display the location on a map
                                #map_sby = folium.Map(location=[float(Latitude), float(Longitude)], zoom_start=15)
                                #tooltip = location
                                #folium.Marker([float(Latitude), float(Longitude)], popup=location, tooltip=tooltip).add_to(map_sby)
                                #folium_static(map_sby)
                            else:
                                print("**GPS COORDINATES NOT AVAILABLE!!!**")
                        else:
                            print("**NO METADATA AVAILABLE!!!**")
                except Exception as e:
                    print(f"Error occurred during Metadata analysis: {e}")
                
                #img_metadata_status logic to be added

                print("QR code and OCR Analysis:\n")
                img_ocr_summary = []
        
                if file is not None:
                    #path_name = imageio.imread(path)
                    #with st.expander("**Uploaded Invoice**"):
                        #st.write("**FileName :** ", path.name)
                        #st.image(path)
                    with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
                        #f.write(file_path.getbuffer())
                        # Open the file using Pillow's Image module
                        image = Image.open(file_path)

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
                            print("QR Code Detected!")
                            print('Scanning QR...')
                            time.sleep(1)
                            print("Available Data In QR Code:",result)
                                

                                # Call checkValues function and display result
                            data_to_check = result  # Replace with actual data to check
                            invoice_num = inv_data['invoice_no']
                            value_to_check = invoice_num  # Replace with actual value to check
                            result_match , check_result = data_ocr.checkValues(data_to_check, value_to_check)
                            invoice_dat = inv_data['invoice_date']
                            date_value_to_check = invoice_dat
                            dates_match , check_date = data_ocr.checkDates(result, date_value_to_check)
                            
                            print("Invoice Number Matching :")
                            print(check_result)
                            print("Invoice Date Matching :")
                            print(check_date)
                            
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
                            print("**Total Amount spend Check :**")
                            print(result_check_amount)
                            if result_match and dates_match and amount_match:
                                img_ocr_summary.append({"Status": "No Tampering detected"})
                            else:
                                img_ocr_summary.append({"Status": "Tampering detected"})
                            print(img_ocr_summary)
                        else:
                            print("No QR Code found in the image.")
                    os.remove(f.name)
                    img_ocr_status = any(result["Status"] == "Tampering detected" for result in img_ocr_summary)
                    


                print("Detectron2 Object Detection:")
                img_objectdetection_summary = []
                # Save the image temporarily to a file
                with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as temp_file:
                    image.save(temp_file, format='JPEG')
                    temp_file_path = temp_file.name

                # Now apply the detector on the new temporary file path
                detector = Detector(model_type="objectDetection")
                out = detector.onImage(temp_file_path) 
                img_objectdetection_summary.append({"Status": out})    
                tout="INVOICE IS TAMPERED"
                if out==tout:
                    print(out)
                    print("Proccesed Invoice")
                    with open(file_path,mode = "wb") as f: 
                        #detector.onImage(f.name)
                        img_ = Image.open("result.jpg")
                        ob_file_path = os.path.join(RESULTS_DIR, "Object Detection result.jpg")
                        cv2.imwrite(ob_file_path, np.array(img_))
                        #st.image(img_, caption='Proccesed Image.')
                else:
                    print(out)
                os.remove(temp_file.name)
                img_objectdetection_status = any(result["Status"] == tout for result in img_objectdetection_summary)


                print("Detectron2 Instance Segmentation:")
                img_instanceseg_results_summary = []
                try:
                    segmentor = Segmentor()
                    img_np = np.array(image)
                    outputs = segmentor.inference(img_np)
                    num_instances = len(outputs['instances'])

                    if num_instances == 0:
                        print("No potential tampered regions found.")
                        img_instanceseg_results_summary.append({"Status": "No tampered regions"})  # SN3 added
                    else:
                        print("The invoice appears to be tampered")
                        img_instanceseg_results_summary.append({"Status": "Tampered regions detected"})  # SN3 added
                        out_image = segmentor.output_image(img_np, outputs)
                        print("Processed")
                        is_file_path = os.path.join(RESULTS_DIR, "Instance Segmentation result.jpg")
                        cv2.imwrite(is_file_path, out_image)
                        #st.image(out_image, caption=f'Processed Image - Page {page_num + 1}', use_column_width=True)  # Output 6
                    
                    img_instanceseg_status = any(result["Status"] == "Tampered regions detected" for result in img_instanceseg_results_summary)

                except Exception as e:
                    print(f"Error occurred during instance segmentation analysis: {e}") 
                    
                print("Copy Move Detection")  #SN3 commented
                img_copymove_summary = []

                with NamedTemporaryFile(dir='.', suffix=f'.jpg', delete=False) as f:
                    # Convert the image to RGB (if it's not already)
                    # Save the image to the temporary file
                    image.save(f.name, format='JPEG')

                    try:
                        file_image_path = f.name
                        file_image = cv2.imread(file_image_path)
                        detect = Detect(file_image)

                        key_points, descriptors = detect.siftDetector()
                        forgery, status = detect.locateForgery(eps=20, min_sample=2)
                        
                        if status:
                            img_copymove_summary.append({"Status": "TAMPERING FOUND"})
                        else:
                            img_copymove_summary.append({"Status": "NO TAMPERING FOUND"})
                        
                        # Print or log the processed image if needed
                        # Here we simply print the status and save the image if tampering is found
                        print(status)
                        if status:
                            cm_file_path = os.path.join(RESULTS_DIR, "copymove_result.jpg")
                            cv2.imwrite(f'forgery_detected.jpg', forgery)

                    except Exception as e:
                        print(f"Error occurred during copy move detection: {e}")
                #copy_move status logic to be added
                img_copymove_status = any(result["Status"] == "TAMPERING FOUND" for result in img_copymove_summary)
                os.remove(f.name)
                    
                summary_data = []
                tampering_modules = []
                """if img_metadata_status:
                    tampering_modules.append("Metadata Analysis")"""
                if img_ocr_status:
                    tampering_modules.append("QR and OCR Analysis")
                if img_objectdetection_status:
                    tampering_modules.append("Object Detection Analysis")
                if img_instanceseg_status:
                    tampering_modules.append("Instance Segmentation Analysis")
                if img_copymove_status:  # need to create logic to flag
                     tampering_modules.append("Copy-Move Detection")
                
                
                #if qr_status:
                    #tampering_modules.append("QR Code Analysis")
                
                # Joining the detected modules with HTML line break
                tampering_modules_string = ",".join(tampering_modules) if tampering_modules else "None"
                summary_data.append({
                    "Sl. No.": 1,
                    "File ID": file_id,
                    "Tampering detected modules": tampering_modules_string,
                    "Alert": "Yes" if any([img_ocr_status, img_objectdetection_status, img_instanceseg_status, img_copymove_status]) else "No"
                })

                print(summary_data)

                json_file_path = os.path.join(RESULTS_DIR, 'summary_data.json')
                with open(json_file_path, 'w') as json_file:
                    json.dump(summary_data, json_file, indent=4)
                        


    else:
        print("No supported files found in the 'uploads' folder.")

    

if __name__ == '__main__':
		main()
