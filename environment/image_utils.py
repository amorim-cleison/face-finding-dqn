import cv2
import numpy as np
from .model import Image, Base

x, y = all_coordinates = (0, 1)
width, height = all_dimensions = (0, 1)
left, top, right, bottom = all_bounds = (0, 1, 2, 3)


def __validate_bounds(bounds: tuple):
    assert len(bounds) == len(all_bounds), "Inconsistent bounds"


def empty_bounds():
    return tuple(None for _ in all_bounds)


def empty_coordinates():
    return tuple(None for _ in all_coordinates)


def empty_dimensions():
    return tuple(None for _ in all_dimensions)


def to_bounds(position: tuple, size: tuple):
    bounds = list(empty_bounds())
    bounds[left], bounds[top] = position[x], position[y]
    bounds[right], bounds[bottom] = sum(position, size)
    return tuple(bounds)


def to_position_size(bounds: tuple):
    __validate_bounds(bounds)
    position = list(empty_coordinates())
    position[x], position[y] = bounds[left], bounds[top]
    size = list(empty_dimensions())
    size[width], size[height] = sub((bounds[right], bounds[bottom]), position)
    return tuple(position), tuple(size)


def get_size(bounds: tuple):
    __validate_bounds(bounds)
    return (bounds[right] - bounds[left]), (bounds[bottom] - bounds[top])


def sum(a: tuple, b: tuple) -> tuple:
    return tuple(xa + xb for xa, xb in zip(a, b))


def sub(a: tuple, b: tuple) -> tuple:
    return tuple(xa - xb for xa, xb in zip(a, b))


def mult(a: tuple, b: tuple) -> tuple:
    return tuple(xa * xb for xa, xb in zip(a, b))


def div(a: tuple, b: tuple) -> tuple:
    return tuple(xa / xb for xa, xb in zip(a, b))


def half(a: tuple) -> tuple:
    return div(a, (2, 2))


def center(position: tuple, size: tuple) -> tuple:
    return tuple(int(p + (s / 2)) for p, s in zip(position, size))


def draw_rect(img: np.ndarray, bounds: tuple, color=(255, 0, 0)):
    cv2.rectangle(img, (bounds[left], bounds[top]),
                  (bounds[right], bounds[bottom]), color, 2)


def to_int(a: tuple):
    return tuple(int(round(x)) for x in a)


def resize(img: Image, factor: float):
    assert (0 < factor), "Invalid resize factor: '{}'".format(factor)

    new_size = tuple(int(x * factor) for x in img.size)
    data_resized = cv2.resize(img.data, new_size, interpolation=cv2.INTER_AREA)
    data_resized = np.reshape(data_resized, (*data_resized.shape, 1))
    return Image(data_resized)


def load_img(path: str):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, (*img.shape, 1))
    return img


def plot_img(img, timeout=1):
    cv2.imshow('Environment Visualization', img)
    try:
        cv2.waitKey(timeout)
    except Exception:
        pass


def pad_img(img, margins: tuple):
    __validate_bounds(margins)

    if any(margins):

        def pad_before(margin_a, margin_b):
            return abs(min(margin_a, 0) - min(margin_b, 0))

        def pad_after(margin_a, margin_b):
            return abs(max(margin_a, 0) - max(margin_b, 0))

        pad_left = pad_before(margins[left], margins[right])
        pad_top = pad_before(margins[top], margins[bottom])
        pad_right = pad_after(margins[right], margins[left])
        pad_bottom = pad_after(margins[bottom], margins[top])

        return np.pad(img,
                      pad_width=((pad_top, pad_bottom), (pad_left, pad_right),
                                 (0, 0)),
                      mode="constant")
    else:
        return img


def crop_img(img, bounds: tuple):
    # Correct bounds:
    negatives = np.array(bounds) < 0
    corrected_bounds = np.where(negatives, 0, bounds)

    # Crop image:
    return img[corrected_bounds[top]:corrected_bounds[bottom],
               corrected_bounds[left]:corrected_bounds[right], :]


def crop_and_pad(img: Image, bounds):
    __validate_bounds(img.bounds)
    __validate_bounds(bounds)

    # Verify if crop will exceed bounds:
    exceeds, out_bounds = exceed_bounds(bounds, img.bounds)

    # Crop image:
    data_cropped = crop_img(img.data, bounds)

    if exceeds:
        # Pad image with out bounds:
        data_cropped = pad_img(data_cropped, out_bounds)

    img_cropped = Image(data_cropped)
    assert (img_cropped.size == get_size(bounds)), "Cropped in a wrong size."
    return (img_cropped, exceeds, out_bounds)


def exceed_bounds(target: tuple, bounds: tuple) -> (tuple, bool):
    __validate_bounds(target)
    __validate_bounds(bounds)

    def calc_diff(target, bound_min, bound_max):
        if target > bound_max:
            return target - bound_max
        elif target < bound_min:
            return target - bound_min
        else:
            return 0

    # Diffs calculation:
    diffs = list(empty_bounds())
    diffs[left] = calc_diff(target[left], bounds[left], bounds[right])
    diffs[top] = calc_diff(target[top], bounds[top], bounds[bottom])
    diffs[right] = calc_diff(target[right], bounds[left], bounds[right])
    diffs[bottom] = calc_diff(target[bottom], bounds[top], bounds[bottom])

    # Return tuple (margins (left, top, right, bottom))
    return any(diffs), tuple(diffs)


def intersects(a: tuple, b: tuple) -> bool:
    return (between(b[left], a[left], a[right])
            or between(b[right], a[left], a[right])) and (
                between(b[top], a[top], a[bottom])
                or between(b[bottom], a[top], a[bottom]))


def between(x, a, b):
    return (x >= a) and (x <= b)


def calc_distance(a: tuple, b: tuple) -> tuple:
    diff = np.subtract(a, b)
    return np.linalg.norm(diff)


def denorm_point(norm_point: tuple, ref: Base):
    return sum(mult(ref.size, norm_point), ref.position)


def norm_point(point: tuple, ref: Base):
    return div(sub(point, ref.position), ref.size)


def normalize_size(size: tuple, ref: Base):
    return div(size, ref.size)


def denormalize_size(norm_size: tuple, ref: Base):
    return mult(norm_size, ref.size)
