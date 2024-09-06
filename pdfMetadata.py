import PyPDF2
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from datetime import datetime
import re

class PDFExtractor:
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
