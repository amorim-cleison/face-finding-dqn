import gym
from gym import spaces
import numpy as np
import cv2
from gym.utils import seeding
from .model import Image, Step, View, Base
from .image_utils import draw_rect, to_int, resize, load_img, plot_img, intersects, calc_distance, crop_and_pad, sum, to_bounds, norm_point, denorm_point, center, sub, half, normalize_size, mult, denormalize_size


class PeopleFramingEnv(gym.Env):
    """
    TODO: document environment
    """

    # Constants:
    state_size = (state_width, state_height) = (300, 300)  # Width x height
    step_size_normal = Step(move=(0.20, 0.20), scale=0.20)
    step_size_small = Step(move=(0.01, 0.01), scale=0.01)

    min_scale = 0.10
    max_scale = 2.00

    min_center = 0.0
    max_center = 1.0

    x, y = (0, 1)
    width, height = (0, 1)
    left, top, right, bottom = (0, 1, 2, 3)

    actions = [
        # Normal steps:
        Step("left", move=(-step_size_normal.move[x], 0), scale=0),
        Step("right", move=(+step_size_normal.move[x], 0), scale=0),
        Step("up", move=(0, -step_size_normal.move[y]), scale=0),
        Step("down", move=(0, +step_size_normal.move[y]), scale=0),
        Step("zoom in", move=(0, 0), scale=+step_size_normal.scale),
        Step("zoom out", move=(0, 0), scale=-step_size_normal.scale),

        # Small steps:
        Step("left sm", move=(-step_size_small.move[x], 0), scale=0),
        Step("right sm", move=(+step_size_small.move[x], 0), scale=0),
        Step("up sm", move=(0, -step_size_small.move[y]), scale=0),
        Step("down sm", move=(0, +step_size_small.move[y]), scale=0),
        Step("zoom in sm", move=(0, 0), scale=+step_size_small.scale),
        Step("zoom out sm", move=(0, 0), scale=-step_size_small.scale)
    ]

    reward = {
        "all-ok": 1.0,
        "partially-ok": 0.0,
        "roi-visible": -1.0,
        "otherwise": -2.0
    }

    def __init__(self, img_path: str):
        # Intialize image:
        self.img = self._init_img(img_path)

        # Initialize roi:
        roi, norm_roi = self._init_roi(self.img, True)
        draw_rect(self.img.data, roi.bounds, color=(0, 0, 255))
        self.roi = norm_roi

        # Setup action and observation spaces:
        state_shape = (*self.state_size, self.img.channels)
        self.action_space = self._init_actions(self.actions)
        self.observation_space = self._init_observation(state_shape)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        # Get step:
        step = self.actions[action]

        # Generate state:
        new_view, new_state, new_exceeds = self._generate_state(
            self.img, self.view, step)

        # Save new state:
        self.state = new_state
        self.view = new_view

        # Verify if roi is visible:
        roi_visible = intersects(self.roi.bounds, new_view.bounds)
        factor = self._calculate_factor(self.roi, self.view, self.img)
        reward, done = self._get_reward(*factor, roi_visible, new_exceeds)
        return self.state.data, reward, done

    def get_action_name(self, action):
        return self.actions[action].name

    def reset(self):
        assert (self.img is not None), "Image was not initialized"

        # Calculate size:
        size = self.state_size
        norm_size = normalize_size(size, self.img)

        # Randomize center:
        min_center = (self.img.bounds[self.left], self.img.bounds[self.top])
        max_center = (self.img.bounds[self.right],
                      self.img.bounds[self.bottom])
        center = self.np_random.uniform(low=min_center, high=max_center)
        norm_center = norm_point(center, self.img)

        # Calculate position:
        position = sub(center, half(size))
        norm_position = norm_point(position, self.img)

        # Randomize scale:
        scale = self.np_random.uniform(low=self.min_scale, high=self.max_scale)

        # Generate and save state:
        new_view = View(norm_position, norm_center, norm_size, scale)
        _, new_state, _ = self._generate_state(self.img, new_view, Step())

        # Save state and view:
        self.state = new_state
        self.view = new_view
        return self.state.data

    def render(self):
        plot_img(self.state.data)

    def _init_actions(self, actions):
        return spaces.Discrete(len(actions))

    def _init_observation(self, state_shape):
        return spaces.Box(low=0, high=255, shape=state_shape, dtype=np.uint8)

    def _init_img(self, path: str) -> Image:
        data = load_img(path)
        return Image(data, (0, 0))

    def _init_roi(self, img: Image, rect=False):
        face_detector = cv2.CascadeClassifier(
            "data/model/haarcascade_frontalface_default.xml")
        detected_faces = face_detector.detectMultiScale(img.data)
        face = self._select_face(detected_faces)
        assert (face is not None), "No face was selected"

        # ROI:
        roi = self._define_roi(face)

        # Normalized ROI:
        norm_position = norm_point(roi.position, img)
        norm_size = normalize_size(roi.size, img)
        norm_center = norm_point(roi.center, img)
        norm_roi = Base(norm_position, norm_center, norm_size)

        return roi, norm_roi

    def _select_face(self, faces):
        return faces[0]

    def _define_roi(self, face):
        (x, y, w, h) = face
        position, size = (x, y), (w, h)

        new_size = [None, None]
        new_size[self.width] = size[self.width] * 3
        new_size[self.height] = size[self.height] * 3

        new_position = [None, None]
        new_position[self.x] = position[self.x] - size[self.width]
        new_position[self.y] = position[self.y]

        new_center = center(new_position, new_size)

        return Base(new_position, new_center, new_size)

    def _generate_state(self, img: Image, view: View, step: Step):
        # Update scale:
        scale = view.scale + step.scale
        scale = max(scale, self.min_scale)
        scale = min(scale, self.max_scale)

        # Update image:
        img_resized = resize(img, scale)

        # Update size:
        size = self.state_size
        norm_size = normalize_size(size, img_resized)

        # Update center:
        norm_center_x, norm_center_y = sum(view.center, step.move)
        norm_center_x = max(norm_center_x, self.min_center)
        norm_center_x = min(norm_center_x, self.max_center)
        norm_center_y = max(norm_center_y, self.min_center)
        norm_center_y = min(norm_center_y, self.max_center)
        norm_center = (norm_center_x, norm_center_y)
        center = to_int(denorm_point(norm_center, img_resized))

        # Update position:
        position = sub(center, half(size))
        norm_position = norm_point(position, img_resized)

        # Create view:
        view = View(norm_position, norm_center, norm_size, scale)

        # Create image (crop to view):
        bounds = to_int(to_bounds(position, size))
        img_cropped, exceeds, _ = crop_and_pad(img_resized, bounds)
        assert (img_cropped.shape == self.observation_space.shape
                ), "Generated state with wrong dimensions."
        return view, img_cropped, exceeds

    def _get_reward(self, side, dist, roi_visible, out_of_image):
        # zoom_ok = (0.8 < side <= 1)
        # dist_ok = (dist <= 0.1)
        # done = False
        zoom_ok = (side <= 0.2)
        dist_ok = (dist <= 0.1)

        # if zoom_ok and dist_ok:
        #     result = "all-ok"
        #     done = True
        # elif zoom_ok or dist_ok:
        #     result = "partially-ok"
        # elif roi_visible:
        #     result = "roi-visible"
        # else:
        #     result = "otherwise"
        # return self.reward[result], done

        if zoom_ok and dist_ok:
            return (1, True)
        else:
            return (-(side + dist), False)

    def _calculate_factor(self, roi: Base, view: View, img: Image):
        # Calculate "side":
        roi_size = denormalize_size(mult(roi.size, (view.scale, view.scale)),
                                    img)
        side = max(roi_size[self.width] / self.state_size[self.width],
                   roi_size[self.height] / self.state_size[self.height])
        side = calc_distance(side, 1)

        # Calculate "dist":
        dist = calc_distance(roi.center, view.center)

        return side, dist
