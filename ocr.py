import easyocr
import pandas as pd
import pytesseract
import cv2
import re
from difflib import SequenceMatcher

class ocr:
    def __init__(self, image):
        self.image = image
        self.keywords = [
            'Calories', 'Total Fat', 'Saturated Fat', 'Trans Fat', 
            'Cholesterol', 'Sodium', 'Total Carbohydrate', 
            'Dietary Fiber', 'Total Sugars', 'Protein'
        ]
        self.extracted_values = {}
        self.ocr_draft()

    def ocr_draft(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Optional: Apply thresholding to enhance text visibility
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR on the entire image
        ocr_text = pytesseract.image_to_string(thresh, config='--psm 6')

        # Print raw OCR output (for debugging)
        print("Raw OCR Output:")
        print(ocr_text)

        # Define regex patterns for each nutritional value
        patterns = {
            'Calories': r'Calories\s*(\d+)',
            'Total Fat': r'Total Fat\s*([\d.]+)\s*g',
            'Saturated Fat': r'Saturated Fat\s*([\d.]+)\s*g',
            'Trans Fat': r'Trans Fat\s*([\d.]+)\s*g',
            'Cholesterol': r'Cholesterol\s*([\d.]+)\s*mg',
            'Sodium': r'Sodium\s*([\d.]+)\s*mg',
            'Total Carbohydrate': r'Total Carbohydrate\s*([\d.]+)\s*g',
            'Dietary Fiber': r'Dietary Fiber\s*([\d.]+)\s*g',
            'Total Sugars': r'Total Sugars\s*([\d.]+)\s*g',
            'Protein': r'Protein\s*([\d.]+)\s*g',
        }

        # Extract values based on the defined patterns
        for key, pattern in patterns.items():
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match:
                extracted_value = match.group(1)
                self.extracted_values[key] = self.clean_value(extracted_value, key)

        # Display the extracted values
        print("\nExtracted Nutrition Facts:")
        for key, value in self.extracted_values.items():
            print(f"{key}: {value}")

    def clean_value(self, value, key):
        """
        Checks if the extracted value resembles a keyword and replaces it if necessary.
        """
        # Check if the value is numeric, if not, compare it to the keywords
        if not value.isdigit():
            for keyword in self.keywords:
                # Check similarity between the extracted value and the keyword
                similarity = SequenceMatcher(None, value.lower(), keyword.lower()).ratio()
                if similarity > 0.5:  # Threshold for similarity; adjust as needed
                    return keyword
        return value
