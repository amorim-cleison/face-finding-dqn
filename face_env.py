import gym
from gym import spaces
import numpy as np
import cv2
import math
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
        self.state_width = 350  # Width of images
        self.state_height = 200  # Height of images

        self.tau = 0.0625

        self.face_detector = cv2.CascadeClassifier(
            "data/model/haarcascade_frontalface_default.xml")

        self.img = self._load_img(img_path)

        self.reward = {"success": 1, "out_of_range": -1, "otherwise": -0.05}

        self.viewer = None

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
            shape=(self.state_height, self.state_width, 3),
            dtype=np.uint8)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # TODO: verificar isto
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        # Take action over position:
        self.position = self._sum(self.position, self.actions[action])
        self.state = self._crop_new_state(self.position)

        # Detect face:
        faces = self._detect_face(self.state)
        fn_result = None

        # If faces were found:
        if len(faces) > 0:
            (face_x, face_y, face_width, face_height) = faces[0]
            face_center = (face_x + (face_width / 2),
                           face_y + (face_height / 2))
            state_center = (self.state_width / 2, self.state_height / 2)

            d = self._distance(state_center, face_center)
            fn_result = (d / (self.state_width + self.state_height))

        reward, done = self._calculate_reward(faces, fn_result)
        return np.array(self.state), reward, done, {}

    def reset(self):
        img_heigth, img_width, _ = self.img.shape
        max_x = img_width - self.state_width
        max_y = img_heigth - self.state_height

        random_pos = self.np_random.uniform(low=(0, 0), high=(max_x, max_y))
        self.position = tuple(int(x) for x in random_pos)
        self.state = self._crop_new_state(self.position)
        return self.state

    def render(self):
        self._plot_img(self.state)

    def _load_img(self, path):
        return cv2.imread(path)

    def _detect_face(self, img, plot_result=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray)

        if plot_result:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self._plot_img(img)
        return faces

    def _crop_new_state(self, new_position: tuple):
        (x, y) = new_position
        return self.img[y:y + self.state_height, x:x + self.state_width]

    def _plot_img(self, img):
        cv2.imshow('Image Visualization', img)
        try:
            cv2.waitKey(1)
        except Exception:
            pass

    def _sum(self, a: tuple, b: tuple) -> tuple:
        return [sum(x) for x in zip(a, b)]

    def _distance(self, a: tuple, b: tuple) -> tuple:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)

    def _calculate_reward(self, faces, fn_result):
        if len(faces) > 0:
            if fn_result <= self.tau:
                result = "success"
                done = True
            else:
                result = "otherwise"
                done = False
        else:
            result = "out_of_range"
            done = True
        return (self.reward[result], done)
