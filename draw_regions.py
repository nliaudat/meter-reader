###### USAGE : 
    # Replace "alg.jpg" with the path to your image.
    # Run the script.
    # Draw regions on the image by clicking and dragging the mouse.
    # Press q to quit and save the regions to regions.txt.

import cv2

# Global variables
regions = []
drawing = False
ix, iy = -1, -1

# Mouse callback function
def draw_region(event, x, y, flags, param):
    global ix, iy, drawing, regions

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        regions.append((ix, iy, x, y))  # Save the region (x1, y1, x2, y2)
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)  # Draw the rectangle
        cv2.imshow("Image", img)

# Load an image
image_path = "sample.jpg"  # Replace with your image path
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Create a window and bind the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_region)

# Display the image and wait for user input
while True:
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit and save the regions
    if key == ord('q'):
        break

# Save the regions to a text file
output_file = "regions.txt"
with open(output_file, "w") as f:
    for region in regions:
        f.write(f"{region[0]},{region[1]},{region[2]},{region[3]}\n")

print(f"Regions saved to {output_file}")

# Clean up
cv2.destroyAllWindows()