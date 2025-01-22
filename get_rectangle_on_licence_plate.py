import cv2
import os

# Path to the Haar Cascade XML file for license plate detection
cascade_path = "./haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print("Error: Failed to load Haar Cascade XML file. Check the path.")
    exit()

def detect_license_plate(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect license plates
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw green rectangles around detected plates
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Save the result to the current working directory
    output_path = os.path.join(os.getcwd(), "output_license_plate.jpg")
    cv2.imwrite(output_path, image)
    print(f"Image with detected license plate saved to: {output_path}")

# Path to the input image
image_path = "./car_image.png"
detect_license_plate(image_path)
