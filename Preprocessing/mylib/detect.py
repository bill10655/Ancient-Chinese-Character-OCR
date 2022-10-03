import cv2
import numpy as np


def resize(image):
    if max(image.shape) > 1200:
        if image.shape[0] > image.shape[1]:
            image = cv2.resize(
                image, (int(1200 * image.shape[1] / image.shape[0]), 1200)
            )
        else:
            image = cv2.resize(
                image, (1200, int(1200 * image.shape[0] / image.shape[1]))
            )
    return image


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(image):
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )


def morph(image, orientation):
    # closing
    if orientation == "horizontal":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))

    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return result


def denoising(image):
    return cv2.medianBlur(image, 1)


def filter_component(image):
    # init
    area_threshold = 3
    height_threshold = image.shape[0] * 0.2
    width_threshold = image.shape[1] * 0.2

    # find component
    _, image_label, components, _ = cv2.connectedComponentsWithStats(image)

    # draw component
    result = np.zeros(image.shape, dtype=np.uint8)
    for i, comp in enumerate(components):
        if (
            comp[cv2.CC_STAT_WIDTH] < width_threshold
            and comp[cv2.CC_STAT_HEIGHT] < height_threshold
            and comp[cv2.CC_STAT_AREA] > area_threshold
        ):
            result[image_label == i] = 255

    return result


def find_component(image, orientation):
    def find_mean(values):
        # pre-filter
        values = np.array([x for x in values if 3 < x])

        # iteration filter
        for _ in range(3):
            new_values = np.array(
                [x for x in values if values.mean() / 2 < x < values.mean() * 3]
            )
            if len(values) == len(new_values):
                break
            else:
                values = new_values

        return values.mean()

    # find component
    _, _, components, _ = cv2.connectedComponentsWithStats(image)

    # set threshold
    height_max = image.shape[0]
    width_max = image.shape[1]
    height_min = 7
    width_min = 7

    if orientation == "horizontal":
        heights = np.array(
            [component[cv2.CC_STAT_HEIGHT] for component in components[1:]]
        )
        height_mean = find_mean(heights)
        height_max = height_mean * 3
        height_min = height_mean / 2
    else:
        widths = np.array(
            [component[cv2.CC_STAT_WIDTH] for component in components[1:]]
        )
        width_mean = find_mean(widths)
        width_max = width_mean * 3
        width_min = width_mean / 2

    area_min = 10
    shape_rate_max = 50
    fill_rate_min = 0.15

    # draw bounding box
    image_component = np.zeros(image.shape, dtype=np.uint8)
    for rect in components[1:]:
        shape_rate = max(rect[cv2.CC_STAT_WIDTH], rect[cv2.CC_STAT_HEIGHT]) / min(
            rect[cv2.CC_STAT_WIDTH], rect[cv2.CC_STAT_HEIGHT]
        )
        fill_rate = rect[cv2.CC_STAT_AREA] / (
            rect[cv2.CC_STAT_WIDTH] * rect[cv2.CC_STAT_HEIGHT]
        )
        if (
            width_max > rect[cv2.CC_STAT_WIDTH] > width_min
            and height_max > rect[cv2.CC_STAT_HEIGHT] > height_min
            and rect[cv2.CC_STAT_AREA] > area_min
            and shape_rate < shape_rate_max
            and fill_rate > fill_rate_min
        ):
            p1 = (rect[0], rect[1])
            p2 = (rect[0] + rect[2] - 1, rect[1] + rect[3] - 1)
            cv2.rectangle(image_component, p1, p2, 255, thickness=1)

    return image_component


def draw_bounding_box(image, image_contour):
    # find contours
    _, contours, _ = cv2.findContours(
        image_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # draw bounding box
    result = image.copy()
    for contour in contours:
        rect = cv2.boundingRect(contour)
        p1 = (rect[0], rect[1])
        p2 = (rect[0] + rect[2], rect[1] + rect[3])
        cv2.rectangle(result, p1, p2, 255, thickness=1)

    return result


def extract(image, image_contour):
    # find contours
    _, contours, _ = cv2.findContours(
        image_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # extract text region
    text_region = np.zeros(image.shape, dtype=np.uint8)
    for contour in contours:
        rect = cv2.boundingRect(contour)
        text_region[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]] = image[
            rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]
        ]

    # blend image
    result = cv2.addWeighted(image, 0.5, text_region, 0.5, 0)

    return result
