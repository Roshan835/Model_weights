import pymupdf  # PyMuPDF
import cv2
import numpy as np

class FontExtractor:
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
                            
                            results.append((text, x0, y0, x1, y1, page_num))
        
        pdf.close()
        return results

    def draw_boxes(image, boxes):
        image_copy = np.copy(image)
        
        for box in boxes:
            _, x0, y0, x1, y1, _ = box
            cv2.rectangle(image_copy, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        
        return image_copy