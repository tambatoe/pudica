import matplotlib.pyplot as plt
import cv2


def draw_detections(image_path_, detections):
    image_d = cv2.imread(image_path_)
    width = image_d.shape[1]
    height = image_d.shape[0]
    for p in detections:
        xx = int(p[1] * width)
        yy = int(p[0] * height)
        cv2.circle(image_d, (xx, yy), 4, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(image_d, cv2.COLOR_BGR2RGB))
    plt.show()
