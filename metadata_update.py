import streamlit as st
from PIL import Image, TiffImagePlugin, ExifTags, WebPImagePlugin
from PIL.ExifTags import TAGS, GPSTAGS
import folium
from streamlit_folium import st_folium
import PyPDF2
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from io import BytesIO
from datetime import datetime
import re

# Class to handle image metadata extraction
class ImageMetaData:
    exif_data = None
    image = None

    def __init__(self, img_path):
        self.image = Image.open(img_path)
        self.get_exif_data()

    def get_exif_data(self):
        """Extracts EXIF data for supported image formats like JPEG, TIFF, PNG, and WebP"""
        exif_data = {}
        if self.image.format == 'JPEG' or self.image.format == 'PNG':
            info = self.image._getexif()
            if info:
                for tag, value in info.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    if decoded == "GPSInfo":
                        gps_data = {}
                        for t in value:
                            sub_decoded = GPSTAGS.get(t, t)
                            gps_data[sub_decoded] = value[t]
                        exif_data[decoded] = gps_data
                    else:
                        exif_data[decoded] = value
        elif isinstance(self.image, TiffImagePlugin.TiffImageFile):
            # Extract TIFF metadata directly from the image tags
            for tag, value in self.image.tag_v2.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
        elif isinstance(self.image, WebPImagePlugin.WebPImageFile):
            # WebP may store metadata differently, so check the 'info' dictionary
            exif_data = {}
            info = self.image.info  # This contains WebP metadata, if any
            if 'exif' in info:
                exif = Image.Exif(info['exif'])
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data[decoded] = value
            if not exif_data:
                exif_data = info  # Just display the whole info dictionary if no EXIF found
        
        self.exif_data = exif_data
        return exif_data

    def get_if_exist(self, data, key):
        return data.get(key, None)

    def convert_to_degrees(self, value):
        """Helper function to convert GPS coordinates to degrees"""
        d = float(value[0].numerator) / float(value[0].denominator)
        m = float(value[1].numerator) / float(value[1].denominator)
        s = float(value[2].numerator) / float(value[2].denominator)
        return d + (m / 60.0) + (s / 3600.0)

    def get_lat_lng(self):
        """Returns the latitude and longitude from the exif data"""
        lat = None
        lng = None
        exif_data = self.get_exif_data()
        if "GPSInfo" in exif_data:
            gps_info = exif_data["GPSInfo"]
            gps_latitude = self.get_if_exist(gps_info, "GPSLatitude")
            gps_latitude_ref = self.get_if_exist(gps_info, "GPSLatitudeRef")
            gps_longitude = self.get_if_exist(gps_info, "GPSLongitude")
            gps_longitude_ref = self.get_if_exist(gps_info, "GPSLongitudeRef")

            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = self.convert_to_degrees(gps_latitude)
                if gps_latitude_ref != "N":
                    lat = -lat
                lng = self.convert_to_degrees(gps_longitude)
                if gps_longitude_ref != "E":
                    lng = -lng
        return lat, lng

# Class to handle PDF extraction
class PDFExtractor:
    
    # Function to extract text from PDF
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

    # Function to extract metadata from PDF
    def extract_metadata_from_pdf(self, file):
        file.seek(0)  # Ensure the file pointer is at the beginning
        parser = PDFParser(file)
        doc = PDFDocument(parser)
        return doc.info

    # Function to normalize date formats
    def extract_year_month_day(self, date_strings):
        date_formats = [
            "%Y%m%d%H%M%S%z",  # Original format
            "%d.%m.%Y",        # dd.mm.yyyy format
            "%b %d, %Y",       # Dec 28, 2022
            "%Y-%m-%d",        # yyyy-mm-dd
            "%m/%d/%Y",        # mm/dd/yyyy
            "%d/%m/%Y",        # dd/mm/yyyy
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

    # Function to extract dates from PDF text
    def extract_date_from_text(self, pdf_text):
        date_patterns = [
            r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b',  # dd.mm.yyyy
            r'\b(\d{4}-\d{2}-\d{2})\b',  # yyyy-mm-dd
            r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',  # Month dd, yyyy
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # mm/dd/yyyy or dd/mm/yyyy
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, pdf_text)
            if date_match:
                return date_match.group(1)
        return None

# Streamlit app
st.title("File Metadata Extractor")

# File uploader
uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "pdf"])

if uploaded_file:
    file_type = uploaded_file.type

    if file_type in ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"]:
        # Handle image metadata
        st.subheader("Image Metadata Extraction")
        img_meta = ImageMetaData(uploaded_file)
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Display general EXIF data
        st.subheader("Extracted Metadata")
        exif_data = img_meta.get_exif_data()
        if exif_data:
            for tag, value in exif_data.items():
                st.write(f"**{tag}**: {value}")
        else:
            st.write("No EXIF data found")

        # Display GPS data if available
        lat, lng = img_meta.get_lat_lng()
        if lat and lng:
            st.subheader("GPS Coordinates")
            st.write(f"**Latitude**: {lat}")
            st.write(f"**Longitude**: {lng}")

            # Display the map with folium
            st.subheader("Map View")
            m = folium.Map(location=[lat, lng], zoom_start=15)
            folium.Marker([lat, lng], popup="Image Location").add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.write("No GPS data found")

    elif file_type == "application/pdf":
        # Handle PDF metadata and content extraction (as in previous version)
        st.subheader("PDF Metadata and Content Extraction")
        pdf_extractor = PDFExtractor()

        # Read the file as bytes to avoid it being closed
        file_data = BytesIO(uploaded_file.read())

        # Extract and display metadata
        metadata = pdf_extractor.extract_metadata_from_pdf(file_data)
        if metadata:
            st.subheader("Extracted Metadata")
            for item in metadata:
                for key, value in item.items():
                    st.write(f"**{key}**: {value}")