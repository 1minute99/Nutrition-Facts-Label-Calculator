import cv2
import numpy as np

class label_detector:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_original = cv2.imread(self.img_path)
        if self.img_original is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        self.img_processed = self.img_original.copy()
        self.contour_max = None
        self.width = 0
        self.height = 0
        self.img_warped = None
        self.run_extraction()

    def run_extraction(self):
        self.convert_to_grayscale()
        self.apply_threshold()
        self.find_rectangular_contours()
        self.get_largest_rectangular_contour()
        self.correct_perspective()

    def convert_to_grayscale(self):
        self.img_processed = cv2.cvtColor(self.img_processed, cv2.COLOR_BGR2GRAY)

    def apply_threshold(self):
        self.img_processed = cv2.adaptiveThreshold(
            self.img_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        self.img_processed = cv2.bitwise_not(self.img_processed)
        self.img_processed = cv2.dilate(self.img_processed, None, iterations=1)

    def find_rectangular_contours(self):
        contours, _ = cv2.findContours(self.img_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_contours = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and peri > 1000:
                rectangular_contours.append(approx)

        self.rectangular_contours = rectangular_contours

    def get_largest_rectangular_contour(self):
        max_area = 0
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_max = contour

    def reorder_points(self, points):
        points = points.reshape((4, 2))
        new_points = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        new_points[0] = points[np.argmin(s)]  # Top-left point
        new_points[2] = points[np.argmax(s)]  # Bottom-right point

        diff = np.diff(points, axis=1)
        new_points[1] = points[np.argmin(diff)]  # Top-right point
        new_points[3] = points[np.argmax(diff)]  # Bottom-left point

        return new_points

    def correct_perspective(self):
        pts_src = self.reorder_points(self.contour_max)

        # Scaling factor to increase resolution
        scale_factor = 2

        self.width = int(scale_factor * max(np.linalg.norm(pts_src[0] - pts_src[1]), np.linalg.norm(pts_src[2] - pts_src[3])))
        self.height = int(scale_factor * max(np.linalg.norm(pts_src[0] - pts_src[3]), np.linalg.norm(pts_src[1] - pts_src[2])))

        pts_dst = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]], dtype="float32")

        t_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        self.img_warped = cv2.warpPerspective(
            self.img_original, 
            t_matrix, 
            (self.width, self.height), 
            flags=cv2.INTER_LANCZOS4  # High-quality interpolation
        )

        # Apply sharpening filter to enhance details
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.img_warped = cv2.filter2D(self.img_warped, -1, kernel)

    def get_final_image(self):
        return self.img_warped
