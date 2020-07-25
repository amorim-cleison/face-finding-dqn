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

    def __init__(self, img_path):
        # self.state_width = 84  # Width of images
        # self.state_height = 84  # Height of images
        self.state_width = 200  # Width of images
        self.state_height = 200  # Height of images

        self.tau = 0.0625

        self.face_detector = cv2.CascadeClassifier(
            "data/model/haarcascade_frontalface_default.xml")

        self.img = self._load_img(img_path)

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
        # TODO: verificar isto
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        faces = None
        fn_result = None

        # Take action over position:
        self.position = self._sum(self.position, self.actions[action])
        out_of_range, _, sizes = self._check_out_of_range(self.position)

        new_state = self._crop_new_state(self.img, self.position)
        new_state = self._pad_img(new_state, sizes)
        self.state = new_state

        # Detect face:
        faces = self._detect_face(self.state)

        # If faces were found:
        if len(faces) > 0:
            (face_x, face_y, face_width, face_height) = faces[0]
            face_center = (face_x + (face_width / 2),
                           face_y + (face_height / 2))
            state_center = (self.state_width / 2, self.state_height / 2)

            d = self._distance(state_center, face_center)
            fn_result = (d / (self.state_height + self.state_width))

        reward, done = self._calculate_reward(out_of_range, faces, fn_result)
        return np.array(self.state), reward, done, {}

    def _pad_img(self, img, sizes):
        if any(sizes > 0):
            (top, left, bottom, right) = sizes
            return np.pad(
                img,
                pad_width=((top, bottom), (left, right), (0, 0)),
                mode="constant")
        else:
            return img

    def _check_out_of_range(self, position):
        # Position bounds:
        left, top = position
        right, bottom = left + self.state_width, top + self.state_height

        # Rectangle bounds:
        rect_left, rect_top = (0, 0)
        rect_right, rect_bottom = rect_left + self.img.shape[
            1], rect_top + self.img.shape[0]

        # Out of bounds calculation:
        sides = np.array(['top', 'left', 'bottom', 'right'])
        diffs = [(top - rect_top), (left - rect_left), (rect_bottom - bottom),
                 (rect_right - right)]
        exceeds = np.array(diffs) < 0
        margins = np.where(exceeds, diffs, 0)

        # Return tuple (out_of_range, sides, margins (top, left, bottom, right))
        return (any(exceeds), sides[exceeds], abs(margins))

    def reset(self):
        img_heigth, img_width, _ = self.img.shape
        max_x = img_width - self.state_width
        max_y = img_heigth - self.state_height

        random_pos = self.np_random.uniform(low=(0, 0), high=(max_x, max_y))
        self.position = tuple(int(x) for x in random_pos)
        self.state = self._crop_new_state(self.img, self.position)
        return self.state

    def render(self):
        self._plot_img(self.state)

    def _load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        return img

    def _detect_face(self, img, plot_result=False):
        faces = self.face_detector.detectMultiScale(img)

        if plot_result:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self._plot_img(img)
        return faces

    def _crop_new_state(self, img, new_position: tuple):
        (right, top) = self._sum(new_position,
                                 (self.state_width, self.state_height))

        negatives = np.array(new_position) < 0
        (left, bottom) = np.where(negatives, 0, new_position)

        return img[bottom:top, left:right]

    def _plot_img(self, img):
        cv2.imshow('Image Visualization', img)
        try:
            cv2.waitKey(1)
        except Exception:
            pass

    def _sum(self, a: tuple, b: tuple) -> tuple:
        return [sum(x) for x in zip(a, b)]

    def _distance(self, a: tuple, b: tuple) -> tuple:
        diff = np.subtract(a, b)
        return np.linalg.norm(diff)

    def _calculate_reward(self, out_of_range, faces, fn_result):
        done = False

        if out_of_range:
            result = "out_of_range"
        elif len(faces) > 0 and fn_result <= self.tau:
            result = "success"
            done = True
        else:
            result = "otherwise"
        return (self.reward[result], done)
