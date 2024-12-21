import math
import torch
import numpy as np
from collections import deque
from utils import core

class StateSpace:
    def __init__(self, args):
        self.feature_history = args.feature_history
        self.glucose = deque(self.feature_history*[0], self.feature_history)
        self.insulin = deque(self.feature_history*[0], self.feature_history)
        self.glucose_max = args.glucose_max
        self.glucose_min = args.glucose_min
        self.insulin_max = args.insulin_max
        self.insulin_min = args.insulin_min
        self.t_meal = args.t_meal
        self.use_carb_announcement = args.use_carb_announcement
        self.mealAnnounce = args.use_meal_announcement
        self.todAnnounce = args.use_tod_announcement

        if not self.mealAnnounce and not self.todAnnounce:  # only ins and glucose
            self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)
        elif self.todAnnounce:
            self.meal_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.tod_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.hc_iob_20 = deque(self.feature_history * [0], self.feature_history)
            self.hc_iob_60 = deque(self.feature_history * [0], self.feature_history)
            self.hc_iob_120 = deque(self.feature_history * [0], self.feature_history)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.tod_announcement_arr,
                                   self.hc_iob_20, self.hc_iob_60, self.hc_iob_120), axis=-1).astype(np.float32)
        elif self.use_carb_announcement:
            self.meal_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.carb_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.carb_announcement_arr), axis=-1).astype(np.float32)
        else:
            self.meal_announcement_arr = deque(self.feature_history * [0], self.feature_history)
            # self.meal_type_arr = deque(self.feature_history * [0], self.feature_history)
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr), axis=-1).astype(np.float32)

    def update(self, cgm=0, ins=0, meal=0, hour=0, meal_type=0, carbs=0, activity_intensity=0, activity_duration=0):
        cgm = core.linear_scaling(x=cgm, x_min=self.glucose_min, x_max=self.glucose_max)
        ins = core.linear_scaling(x=ins, x_min=self.insulin_min, x_max=self.insulin_max)
        hour = core.linear_scaling(x=hour, x_min=0, x_max=23)  # hour is given 0-23
        t_to_meal = core.linear_scaling(x=meal, x_min=0, x_max=self.t_meal)
        carbs = core.linear_scaling(x=carbs, x_min=0, x_max=120)

        # Dynamically handle activity features
        activity_intensity = core.linear_scaling(x=activity_intensity, x_min=0,
                                                 x_max=50)  # Scale activity intensity (e.g., 0 to 10)
        activity_duration = core.linear_scaling(x=activity_duration, x_min=0,
                                                x_max=120)  # Scale duration (e.g., max 120 minutes)

        # Append new glucose and insulin values
        self.glucose.append(cgm)
        self.insulin.append(ins)

        # Handcrafted features for insulin-on-board
        ins_20 = sum(self.state[-4:, 1])  # last 20 min
        ins_60 = sum(self.state[-12:, 1])  # last 60 min
        ins_120 = sum(self.state[-24:, 1])  # last 120 min

        if not self.mealAnnounce and not self.todAnnounce:
            self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(np.float32)
        elif self.todAnnounce:
            self.meal_announcement_arr.append(t_to_meal)
            self.tod_announcement_arr.append(hour)
            self.hc_iob_20.append(ins_20)
            self.hc_iob_60.append(ins_60)
            self.hc_iob_120.append(ins_120)

            # New activity features added to the state.
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.tod_announcement_arr,
                                   self.hc_iob_20, self.hc_iob_60, self.hc_iob_120,
                                   [activity_intensity] * self.feature_history,  # Repeat for feature history
                                   [activity_duration] * self.feature_history), axis=-1).astype(np.float32)
        elif self.use_carb_announcement:
            self.meal_announcement_arr.append(t_to_meal)
            self.carb_announcement_arr.append(carbs)

            # Activity features for carb announcement case
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr, self.carb_announcement_arr,
                                   [activity_intensity] * self.feature_history,
                                   [activity_duration] * self.feature_history), axis=-1).astype(np.float32)
        else:
            self.meal_announcement_arr.append(t_to_meal)

            # Activity features for simplistic state
            self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr,
                                   [activity_intensity] * self.feature_history,
                                   [activity_duration] * self.feature_history), axis=-1).astype(np.float32)

        # Handcrafted features now include activities
        handcraft_features = [hour, activity_intensity, activity_duration, ins_20, ins_60, ins_120]
        return self.state, handcraft_features
