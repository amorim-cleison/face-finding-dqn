import gym
from gym import spaces
import numpy as np
import cv2
from gym.utils import seeding


class FaceEnvironment(gym.Env):
    """
    FIXME: corrigir
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 20
    """

    def __init__(self, img_path: str, reset_to_face_visible: bool):
        self.face_detector = cv2.CascadeClassifier(
            "data/model/haarcascade_frontalface_default.xml")

        self.img = self._load_img(img_path)
        self.img_position = (0, 0)
        self.img_size = self.img.shape[0:2][::-1]
        self.img_channels = self.img.shape[2]

        faces = self._detect_faces(self.img)
        self.face_bounds = self._choose_face(faces)
        self._draw_rect(self.img, self.face_bounds)
        self._plot_img(self.img)
        self.reset_to_face_visible = reset_to_face_visible

        self.tau = 0.0625
        # self.state_width = 84  # Width of images
        # self.state_height = 84  # Height of images
        self.state_width = 200  # Width of images
        self.state_height = 200  # Height of images
        self.state_size = (self.state_width, self.state_height)
        self.state_shape = (self.state_height, self.state_width,
                            self.img_channels)

        self.reward = {"success": 1, "out_of_range": -1, "otherwise": -0.05}

        # Calculate step size:
        self.num_tiles_x = 20
        self.num_tiles_y = 20
        self.step_size_x = int(self.img.shape[1] / self.num_tiles_x)
        self.step_size_y = int(self.img.shape[0] / self.num_tiles_y)

        # Setup action space;
        self.actions = {
            0: (0, self.step_size_x),
            1: (0, -self.step_size_x),
            2: (0, self.step_size_y),
            3: (0, -self.step_size_y)
        }
        self.action_space = spaces.Discrete(len(self.actions))

        # Setup observation space:
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.state_height, self.state_width, self.img.shape[2]),
            dtype=np.uint8)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        fn_result = None

        # Take action over position:
        self.position = self._sum(self.position, self.actions[action])

        # Generate new state:
        new_state, new_state_bounds, exceeds, _ = self._crop_and_correct_img(
            self.img, self.position, self.state_size)
        assert (new_state.shape == self.state_shape
                ), "Generated state with wrong dimensions."

        self.state = new_state

        # Verify face intersection:
        intersects_face = self._intersects(self.face_bounds, new_state_bounds)

        # If faces were found:
        if intersects_face:
            d = self._center_distance(self.face_bounds, new_state_bounds)
            fn_result = (d / (self.state_height + self.state_width))

        reward, done = self._calculate_reward(exceeds, intersects_face,
                                              fn_result)
        return np.array(self.state), reward, done, {}

    def _pad_img(self, img, sizes):
        if any(sizes > 0):
            (left, top, right, bottom) = sizes
            return np.pad(
                img,
                pad_width=((top, bottom), (left, right), (0, 0)),
                mode="constant")
        else:
            return img

    def _detect_faces(self, img):
        faces = self.face_detector.detectMultiScale(img)

        # Map coordinates (x, y, w, h) to bounds (left, top, right, bottom):
        return [(x, y, (x + w), (y + h)) for (x, y, w, h) in faces]

    def _draw_rect(self, img, bounds: tuple):
        left, top, right, bottom = bounds
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

    def _choose_face(self, faces):
        return faces[0]

    def _intersects(self, a: tuple, b: tuple) -> bool:
        a_left, a_top, a_right, a_bottom = a
        b_left, b_top, b_right, b_bottom = b
        intersects = (b_left < a_right
                      and b_top < b_bottom) or (b_right > a_left
                                                and b_bottom > a_top)
        return intersects

    def _center(self, bounds):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        return (left + (width / 2), top + (height / 2))

    def _crop_and_correct_img(self, img, position: tuple, size: tuple):
        # Get bounds (left, top, right, bottom):
        left, top = position
        right, bottom = self._sum(position, size)
        bounds = (left, top, right, bottom)

        img_left, img_top = self.img_position
        img_right, img_bottom = self._sum(self.img_position, self.img_size)
        img_bounds = (img_left, img_top, img_right, img_bottom)

        # Verify if crop will exceed bounds:
        out_bounds, exceeds = self._get_out_bounds(bounds, img_bounds)

        # Crop image:
        new_img = self._crop_img(img, bounds)

        # Pad image with out bounds:
        new_img = self._pad_img(new_img, out_bounds)

        assert (new_img.shape[:2] == size), "Cropped in a wrong size."
        return (new_img, bounds, exceeds, out_bounds)

    def _crop_img(self, img, bounds: tuple):
        # Correct bounds:
        negatives = np.array(bounds) < 0
        corrected_bounds = np.where(negatives, 0, bounds)
        (left, top, right, bottom) = corrected_bounds

        # Crop image:
        new_img = img[top:bottom, left:right]
        return new_img

    def _get_out_bounds(self, target: tuple, bounds: tuple) -> (tuple, bool):
        left, top, right, bottom = target
        bound_left, bound_top, bound_right, bound_bottom = bounds

        # Diffs calculation:
        diff_left = (left - bound_left)
        diff_top = (top - bound_top)
        diff_right = (bound_right - right) if (right > 0) else (
            right - bound_left)
        diff_bottom = (bound_bottom - bottom) if (bottom > 0) else (
            bottom - bound_top)
        diffs = [diff_left, diff_top, diff_right, diff_bottom]

        exceeds = np.array(diffs) < 0
        margins = np.where(exceeds, diffs, 0)

        # Return tuple (margins (left, top, right, bottom))
        return abs(margins), any(exceeds)

    def reset(self):
        if self.reset_to_face_visible:
            (face_left, face_top, face_right, face_bottom) = self.face_bounds
            min_x = face_right - self.state_width
            min_y = face_bottom - self.state_height
            max_x, max_y = face_left, face_top
        else:
            img_left, img_top = self.img_position
            img_right, img_bottom = self._sim(self.img_position, self.img_size)
            min_x, min_y = img_left, img_top
            max_x = img_right - self.state_width
            max_y = img_top - self.state_height

        random_pos = self.np_random.uniform(
            low=(min_x, min_y), high=(max_x, max_y))

        self.position = tuple(int(x) for x in random_pos)
        self.state, _, _, _ = self._crop_and_correct_img(
            self.img, self.position, self.state_size)
        return self.state

    def render(self):
        self._plot_img(self.state)

    def _load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        return img

    def _plot_img(self, img):
        cv2.imshow('Environment Visualization', img)
        try:
            cv2.waitKey(1)
        except Exception:
            pass

    def _sum(self, a: tuple, b: tuple) -> tuple:
        return [sum(x) for x in zip(a, b)]

    def _center_distance(self, a: tuple, b: tuple) -> tuple:
        a_left, a_top, a_right, a_bottom = a
        a_center = self._center(a)

        b_left, b_top, b_right, b_bottom = b
        b_center = self._center(b)

        diff = np.subtract(a_center, b_center)
        return np.linalg.norm(diff)

    def _calculate_reward(self, out_of_range, intersects_face, fn_result):
        done = False

        if out_of_range:
            result = "out_of_range"
        elif intersects_face and fn_result <= self.tau:
            result = "success"
            done = True
        else:
            result = "otherwise"
        return (self.reward[result], done)
