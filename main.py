import cv2
import numpy as np


# Function to calculate the cosine of the angle between vectors from p0 to p1 and from p0 to p2
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype("float"), (p2 - p1).astype("float")
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# Function to find squares in the image
def find_squares(img):
    # Apply Gaussian blur to the image
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Initialize an empty list to hold the squares
    squares = []
    # Iterate over each color channel in the image
    for gray in cv2.split(img):
        # Apply a series of thresholds to the image
        for thrs in range(0, 255, 26):
            if thrs == 0:
                # Apply the Canny edge detection algorithm to the image
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                # Apply dilation to the image
                bin = cv2.dilate(bin, None)
            else:
                # Apply a binary threshold to the image
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            # Find the contours in the binary image
            contours, _hierarchy = cv2.findContours(
                bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            # Iterate over each contour
            for cnt in contours:
                # Calculate the arc length of the contour
                cnt_len = cv2.arcLength(cnt, True)
                # Approximate the contour to a polygon
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                # If the polygon has 4 vertices, its area is greater than 1000, and it is convex
                if (
                    len(cnt) == 4
                    and cv2.contourArea(cnt) > 1000
                    and cv2.isContourConvex(cnt)
                ):
                    # Reshape the contour
                    cnt = cnt.reshape(-1, 2)
                    # Calculate the maximum cosine of the angle between all combinations of three points taken from the contour
                    max_cos = np.max(
                        [
                            angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                            for i in range(4)
                        ]
                    )
                    # If the maximum cosine is less than 0.1, add the contour to the list of squares
                    if max_cos < 0.1:
                        squares.append(cnt)
    # Return the list of squares
    return squares


# Read an image from file
img = cv2.imread("testpic.png")
# Find the squares in the image
squares = find_squares(img)
# Draw the squares on the image
cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
# Display the image
cv2.imshow("squares", img)
# Wait for the user to press any key
cv2.waitKey(0)
# Close the image display
cv2.destroyAllWindows()
