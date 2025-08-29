import json
import os
from typing import Dict
import tempfile
from fpdf import FPDF


def save_json(data: Dict, output_directory: str, output_file_name: str) -> None:
    """
    Saves a given JSON object (Python dictionary) to a specified directory as a file.

    :param data: The JSON content as a Python dictionary.
    :param output_directory: The directory where the output file will be saved.
    :param output_file_name: The name for the output JSON file.
    :return: None
    """
    try:
        # Ensure the output directory exists, if not create it
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created directory: {output_directory}")

        # Path to save the output file
        output_file_path = os.path.join(output_directory, output_file_name)

        # Write the JSON data to the output file
        with open(output_file_path, "w") as outfile:
            json.dump(data, outfile, indent=4)
            print(f"Successfully saved data to {output_file_path}")

    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")


def export_to_pdf(itinerary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    try:
        for line in itinerary_text.split("\n"):
            line = line.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 10, line)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        return temp_file.name
    except Exception as e:
        raise Exception(f"PDF generation failed: {str(e)}")
