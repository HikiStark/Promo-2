import cv2
import numpy as np


# Function to calculate the cosine of the angle between vectors from p0 to p1 and from p0 to p2
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype("float"), (p2 - p1).astype("float")
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# Function to find squares in the image
def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(
                bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if (
                    len(cnt) == 4
                    and cv2.contourArea(cnt) > 10
                    and cv2.isContourConvex(cnt)
                ):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max(
                        [
                            angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                            for i in range(4)
                        ]
                    )
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


# Read an image from file
img = cv2.imread("testpic1.png")

original_width = 2418
original_height = 1773
new_width = 1200

# Calculate the new height while maintaining the aspect ratio
new_height = int((new_width * original_height) / original_width)

# Resize the image
img_resized = cv2.resize(img, (new_width, new_height))

img = img_resized

# Define the source and destination points for the perspective transformation
src_pts = np.float32([[100, 100], [300, 10], [10, 300], [300, 300]])
dst_pts = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])

# # Define the source points as the corners of the rectangle in your image
# src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

# # Define the destination points as the corners of a square in the transformed image
# dst_pts = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])


# Get the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective transformation
img_transformed = cv2.warpPerspective(img, matrix, (1000, 1000))

# Find the squares in the transformed image
squares = find_squares(img_transformed)

# Draw the squares on the transformed image
cv2.drawContours(img_transformed, squares, -1, (0, 255, 0), 3)

# Display the original and transformed images
# cv2.imshow("Original", img)
cv2.imshow("Transformed", img_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
