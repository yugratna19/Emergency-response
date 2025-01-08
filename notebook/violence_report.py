#pip install Pillow reportlab PyPDF2

import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta
from PyPDF2 import PdfReader, PdfWriter

def create_or_append_pdf(image_path):
    # Get the current date for the PDF filename
    current_date = datetime.now().strftime("%Y%m%d")
    pdf_path = f"Violence_Report_{current_date}.pdf"  # Create a filename with the current date

    # Check the last event timestamp
    last_event_time = get_last_event_time()

    # Get current event time
    current_event_time = datetime.now()

    # Check if it's been at least 5 minutes since the last event
    if last_event_time and (current_event_time - last_event_time) < timedelta(minutes=5):
        print("New event did not occur after 5 minutes from the last event.")
        return

    # Update the last event time
    update_last_event_time(current_event_time)

    if os.path.exists(pdf_path):
        # If the PDF already exists, append the new image
        append_image_to_pdf(pdf_path, image_path)
    else:
        # Create a new PDF
        create_pdf(image_path, pdf_path)

def create_pdf(image_path, pdf_path):
    # Create a PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Draw the image on the PDF
    c.drawImage(image_path, 100, 500, width=400, height=300)  # Adjust position and size as needed
    
    # Add the current time to the PDF
    c.drawString(100, 480, f"Violence Happened at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Finalize the PDF
    c.save()
    print(f"New PDF created successfully: {pdf_path}")

def append_image_to_pdf(pdf_path, image_path):
    # Create a temporary PDF for the new image
    temp_pdf_path = "temp.pdf"
    c = canvas.Canvas(temp_pdf_path, pagesize=letter)
    
    # Draw the new image on the temporary PDF
    c.drawImage(image_path, 100, 500, width=400, height=300)  # Adjust position and size as needed
    
    # Add the current time to the temporary PDF
    c.drawString(100, 480, f"Violence Happened at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Finalize the temporary PDF
    c.save()
    
    # Append the temporary PDF to the existing PDF
    writer = PdfWriter()
    
    # Read the existing PDF
    with open(pdf_path, "rb") as existing_pdf:
        reader = PdfReader(existing_pdf)
        for page in reader.pages:
            writer.add_page(page)
    
    # Read the temporary PDF
    with open(temp_pdf_path, "rb") as temp_pdf:
        temp_reader = PdfReader(temp_pdf)
        for page in temp_reader.pages:
            writer.add_page(page)
    
    # Save the updated PDF
    with open(pdf_path, "wb") as updated_pdf:
        writer.write(updated_pdf)
    
    # Remove the temporary file
    os.remove(temp_pdf_path)
    print(f"Image appended to existing PDF: {pdf_path}")

def get_last_event_time():
    """Retrieve the last event time from a file."""
    if os.path.exists("last_event_time.txt"):
        with open("last_event_time.txt", "r") as f:
            timestamp = f.read().strip()
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return None

def update_last_event_time(current_time):
    """Update the last event time to a file."""
    with open("last_event_time.txt", "w") as f:
        f.write(current_time.strftime("%Y-%m-%d %H:%M:%S"))

# Example usage
image_path = r"output\frame_0.jpg"  # Replace with your image path
create_or_append_pdf(image_path)