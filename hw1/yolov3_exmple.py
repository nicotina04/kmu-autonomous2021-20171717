import tensorflow
import cv2
import cvlib
import sys


# Print python environment
def get_env():
    print("Version of tf ->", tensorflow.__version__)
    print("Version of opencv-python ->", cv2.__version__)
    print("Version of python ->", sys.version_info)


if __name__ == "__main__":
    """Print my environment"""
    get_env()

    """Import object_detection (YOLO)"""
    from cvlib.object_detection import draw_bbox

    try:
        """Read an image file and predict"""
        img_path = 'doll.jpeg'
        img_read = cv2.imread(img_path)
        """Detect objects"""
        bbox, label, conf = cvlib.detect_common_objects(img_read)
        """Print and draw the result"""
        print(bbox, label, conf)
        cv2.imwrite('result.jpg', draw_bbox(img_read, bbox, label, conf))
    except AttributeError:
        print("Couldn't find a image file. Exit the program")
