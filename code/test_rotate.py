
import cv2
import random

angle = random.randint(0, 180)
print(angle)
im = cv2.imread('../picset/MD.M108_p3.jpg')

height, width = im.shape[:2]
center = (width / 2, height / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
im = cv2.warpAffine(im, rotation_matrix, (width, height), borderValue=(255, 255, 255))

cv2.imwrite(f'{angle}.jpg', im)





"""

def initial_approx(image):  # EXAMPLE: T17
    global angle
    blurred = cv2.GaussianBlur(image, (5, 5), 5)

    cv2.imwrite('blurred_edition.jpg', blurred)
    ## Edge detection
    thresh = cv2.Canny(blurred, 50, 150, apertureSize=3)

    thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=10)
    thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=10)
    # thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8), iterations=5)
    # thresh = cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=5)

    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 10)

    thresh = skeletonize_image(thresh)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 10)

    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)

    cv2.imwrite('thresh.jpg', thresh)

    # Apply Hough Transform
    lines = cv2.HoughLines(thresh, 4, np.pi / 180, 500)
    # print("Lines:", len(lines))

    cimg = np.copy(image)  # EXAMPLE: T11
    # Draw the lines on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(cimg, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite('lines.jpg', cimg)

        # average of theta values from Hough Transform
        print(lines[:, 0][:, 1])

        theta_radians = find_most_frequent(lines[:, 0][:, 1], 0.05)

        # Convert theta to degrees
        theta_degrees = math.degrees(theta_radians)

        # Calculate the angle of the line with the x-axis
        angle = 90 - theta_degrees

        print("Average Theta (degrees):", round(angle))
        if angle < 0:
            angle = 180 + angle
        print("Error:", abs(round(actual - angle, 2)))


"""