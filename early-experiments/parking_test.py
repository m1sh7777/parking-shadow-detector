import cv2

def a_view_grayscale_image(image_path):
    print(f"loading image from {image_path} and converting to grayscale....")
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}, skipping step A")
        return
    #convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("  Original image shape (height, width, color channels):", image.shape)
    print("  Grayscale image shape (height, width):", gray.shape)
    print("  That's it - color became a single brightness number per pixel.")

    cv2.imshow("Original", image)
    cv2.imshow("Grayscale", gray)
    print("  (Press any key on the image window to continue...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#spotting differences between two images
def b_compare_empty_vs_occupied(empty_path, occupied_path):
    print(f"\nSTEP B: Comparing '{empty_path}' vs '{occupied_path}'...")
    empty = cv2.imread(empty_path)
    occupied = cv2.imread(occupied_path)
 
    if empty is None or occupied is None:
        print("  Missing 'empty.jpg' or 'occupied.jpg'. Skipping Step B.")
        print("  Add both photos (same spot, same angle) and re-run.")
        return

    occupied = cv2.resize(occupied, (empty.shape[1], empty.shape[0])) #makes sure both pics are same size before comparing

    # Convert both images to grayscale
    empty_gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
    occupied_gray = cv2.cvtColor(occupied, cv2.COLOR_BGR2GRAY)

    # Subtract one image from the other.
    # where nothing changed = nearzero (dark) result.
    # where something changed (like a car appearing) = bright result.
    difference = cv2.absdiff(empty_gray, occupied_gray)

    # Count how many pixels changed a proper amount
    _, changed_mask = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(changed_mask)
    total_pixels = changed_mask.size
    percent_changed = (changed_pixels / total_pixels) * 100


    print(f"  {percent_changed:.1f}% of the spot's pixels changed.")

 
 
    # A simple rule: if enough pixels changed, call it "occupied"
    if percent_changed > 15:
        print("  --> Likely OCCUPIED (a lot changed since the empty photo)")
    else:
        print("  --> Likely EMPTY (barely anything changed)")

    cv2.imshow("Empty (reference)", empty)
    cv2.imshow("Occupied (test)", occupied)
    cv2.imshow("Difference (bright = changed)", difference)
    print("  (Press any key on the image window to close...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    a_view_grayscale_image("empty.jpg")
    b_compare_empty_vs_occupied("empty.jpg", "occupied.jpg")
 
    print("\nDone! Next steps once this works:")
    print(" 1. Try it on multiple parking spots in the same photo.")
    print(" 2. Try it with a car partly hidden behind another car or a tree.")
    print(" 3. Notice when the simple 'percent changed' rule gets fooled")
    print("    (e.g. by shadows moving, or lighting changing) = that's exactly")
    print("    the real world problem this whole project is about solving.")
