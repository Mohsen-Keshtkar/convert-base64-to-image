import base64
from imageio import imread
import io
import numpy as np
import cv2
from PIL import Image


class Base64ToImage():

    def read_orientation_matadata(self, image_stream):
        """

        Args:
            image_stream: Binary stream of the image

        Returns: the orientation of image

        """
        # Open the image using Pillow
        image = Image.open(image_stream)

        # Check if the image has Exif data (metadata)
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()

            # Check if the Exif data contains the orientation tag (tag number 274)
            if exif_data and 274 in exif_data:
                # Get the orientation value
                orientation = exif_data[274]

                # Perform actions based on the orientation value
                if orientation == 1:
                    print("Normal (upright) orientation")
                    return 0
                elif orientation == 3:
                    print("Upside down")
                    return 2
                elif orientation == 6:
                    print("Rotated 90 degrees clockwise")
                    return -1
                elif orientation == 8:
                    print("Rotated 90 degrees counter-clockwise")
                    return 1


    def convert_to_gray(self, colored_image: np.ndarray) -> np.ndarray:
        """
        Args:
            colored_image: RGB/BGR image

        Returns: gray-scaled image

        """
        (row, col) = colored_image.shape[0:2]
        gary_img = np.zeros((row, col), dtype=np.uint8)
        for i in range(row):
            for j in range(col):
                gary_img[i, j] = sum(colored_image[i, j]) * 0.33
        return gary_img


    def convert_base64_to_image(self, base64_image: str) -> np.ndarray:
        """This function recieves an image in form of base64 and returns a numpy array (BGR Image) in output.

        Args:
            base64_image (str): base64 string

        Returns:
            np.ndarray: a BGR Image
        """
       
        decoded_img = base64.b64decode(base64_image)
        stream = io.BytesIO(decoded_img)
        converted_img = imread(stream)
        converted_img = self.convert_to_gray(converted_img)
        if self.read_orientation_matadata(stream) == 0 or self.read_orientation_matadata(stream) is None:
            return cv2.cvtColor(converted_img, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.cvtColor(np.rot90(converted_img, k=self.read_orientation_matadata(stream)), cv2.COLOR_GRAY2BGR)