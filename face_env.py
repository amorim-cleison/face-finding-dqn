import gym
from gym import spaces
import numpy as np
import cv2


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

    def __init__(self):
        self.max_steps = 20  # Maximum episode size
        self.num_actions = 8  # Number of actions
        self.width = 84  # Width of images
        self.height = 84  # Height of images

        self.face_detector = cv2.CascadeClassifier(
            'data/haarcascade_frontalface_default.xml')

        self.action_space = spaces.Discrete(self.num_actions)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8)

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        # TODO: #2 buscar face com OpenCV

        done = False  # avaliar dinamicamente se centro face = centro imagem
        reward = 0
        self.state = None  # TODO: #1 carregar imagem

        return np.array(self.state), reward, done, {}

    def reset(self):
        pass

    def render(self):
        pass

    def _load_img(self, path):
        return cv2.imread(path)

    def _detect_face(self, img, plot_result=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray)

        if plot_result:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('img', img)
            cv2.waitKey()

        return faces
