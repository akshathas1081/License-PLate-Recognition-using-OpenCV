import cv2
import numpy as np
import imutils
import easyocr
import tkinter as tk
from tkinter import filedialog, messagebox


# Function to get screen resolution
def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()  # Destroy the root window
    return screen_width, screen_height


# Function to process each frame
def process_frame(frame, reader):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    bfilter = cv2.bilateralFilter(
        gray, 11, 17, 17
    )  # Reduce noise while keeping edges sharp
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[
        :10
    ]  # Get the top 10 contours

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            location = approx
            break

    plate_text = None
    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1 : x2 + 1, y1 : y2 + 1]

        result = reader.readtext(cropped_image)

        if result:
            plate_text = result[0][-2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(
                frame,
                text=plate_text,
                org=(approx[0][0][0], approx[1][0][1] + 60),
                fontFace=font,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            frame = cv2.rectangle(
                frame, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3
            )

    return frame, plate_text


# Function to process an image file
def process_image(image_path, reader):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not open image file.")
        return
    processed_image, plate_text = process_frame(image, reader)
    cv2.imshow("License Plate Detection", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if plate_text:
        messagebox.showinfo(
            "License Plate Detected", f"Detected License Plate: {plate_text}"
        )
    else:
        messagebox.showinfo("License Plate Detection", "No license plate detected.")


# Main function
def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    choice = messagebox.askquestion(
        "Choose Mode",
        "Do you want to process an image file? (Click 'Yes' for image, 'No' for webcam)",
    )

    reader = easyocr.Reader(["en"])  # Initialize OCR reader outside the loop

    if choice == "yes":
        # Ask the user to select an image file
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")],
        )
        if file_path:
            process_image(file_path, reader)
    else:
        # Webcam mode
        screen_width, screen_height = get_screen_resolution()
        window_name = "License Plate Detection"
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        plate_detected = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame.")
                break

            processed_frame, plate_text = process_frame(frame, reader)
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow(window_name, processed_frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                plate_detected = plate_text
                break

        cap.release()
        cv2.destroyAllWindows()

        # Display the detected plate after quitting the loop
        if plate_detected:
            messagebox.showinfo(
                "License Plate Detected", f"Detected License Plate: {plate_detected}"
            )
        else:
            messagebox.showinfo("License Plate Detection", "No license plate detected.")


if __name__ == "__main__":
    main()
