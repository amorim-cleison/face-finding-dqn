import gym
from gym import spaces
import numpy as np
import cv2
from gym.utils import seeding
from .model import Image, Step, View, Base
from .image_utils import draw_rect, to_int, resize, load_img, plot_img, intersects, calc_distance, crop_and_pad, sum, to_bounds, norm_point, denorm_point, center, sub, half, normalize_size
from math import log10


class PeopleFramingEnv(gym.Env):
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

    # Constants:
    state_size = (200, 200)  # Width x height
    step_size = Step((0.1, 0.1), 0.1)

    x, y = (0, 1)
    width, height = (0, 1)
    left, top, right, bottom = (0, 1, 2, 3)

    move_left, move_right, move_up, move_down, zoom_in, zoom_out = (0, 1, 2, 3,
                                                                    4, 5)

    reward = {"success": 1, "visible": -0.05, "otherwise": -1}

    def __init__(self, img_path: str):
        # Intialize image:
        self.img = self._init_img(img_path)

        # Initialize roi:
        roi, norm_roi = self._init_roi(self.img, True)
        draw_rect(self.img.data, roi.bounds)
        self.roi = norm_roi

        # Setup action and observation spaces:
        state_shape = (*self.state_size, 1)
        self.actions, self.action_space = self._init_actions(self.step_size)
        self.observation_space = self._init_observation(state_shape)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

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
        reward, done = self._get_reward(*factor, roi_visible)
        return self.state.data, reward, done

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
        min_scale = 0
        max_scale = 3
        scale = self.np_random.uniform(low=min_scale, high=max_scale)

        # Generate and save state:
        new_view = View(norm_position, norm_center, norm_size, scale)
        _, new_state, _ = self._generate_state(self.img, new_view, Step())

        # Save state and view:
        self.state = new_state
        self.view = new_view
        return self.state.data

    def render(self):
        plot_img(self.state.data)

    def _init_actions(self, step_size: Step):
        actions = {
            self.move_left: Step((-step_size.move[self.x], 0), 0),
            self.move_right: Step((+step_size.move[self.x], 0), 0),
            self.move_up: Step((0, -step_size.move[self.y]), 0),
            self.move_down: Step((0, +step_size.move[self.y]), 0),
            self.zoom_in: Step((0, 0), +step_size.scale),
            self.zoom_out: Step((0, 0), -step_size.scale)
        }
        action_space = spaces.Discrete(len(actions))
        return actions, action_space

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
        norm_roi = Base(norm_position, None, norm_size)

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
        return Base(new_position, None, new_size)

    def _generate_state(self, img: Image, view: View, step: Step):
        # Update scale:
        min_scale = 0.01
        scale = max(view.scale + step.scale, min_scale)

        # Update image:
        img_resized = resize(img, scale)

        # Update size:
        size = self.state_size
        norm_size = normalize_size(size, img_resized)

        # Update center:
        norm_center = sum(view.center, step.move)
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

    def _get_reward(self, side, dist, visible):
        # if factor < 1.5:
        #     result = "success"
        #     done = True
        # elif visible:
        #     result = "visible"
        #     done = False
        # else:
        #     result = "otherwise"
        #     done = False
        # return (self.reward[result], done)

        if (0.9 < side <= 1) and (dist <= 1):
            result = "success"
            done = True
        else:
            result = -(side + dist)
            done = False
        return result, done

    def _calculate_factor(self, roi: Base, view: View, img: Image):
        # Calculate "side":
        side = max(roi.size[self.width] / view.size[self.width],
                   roi.size[self.height] / view.size[self.height])

        # Calculate "dist":
        denorm_view_pos = denorm_point(view.position, img)
        dist = calc_distance(center(roi.position, roi.size), denorm_view_pos)
        dist = log10(dist)

        return side, dist