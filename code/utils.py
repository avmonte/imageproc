import cv2


def mopen(image):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	return cv2.dilate(cv2.erode(image, kernel), kernel)


def mclose(image):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	return cv2.erode(cv2.dilate(image, kernel), kernel)


def mtophat(image):
	return cv2.subtract(image, mopen(image))


def mbothat(image):
	return cv2.subtract(mclose(image), image)
