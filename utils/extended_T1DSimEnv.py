from environments.simglucose.simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from environments.simglucose.simglucose.patient.t1dpatient import T1DPatient
from environments.simglucose.simglucose.sensor.cgm import CGMSensor
from environments.simglucose.simglucose.actuator.pump import InsulinPump
from environments.simglucose.simglucose.controller.base import Action

from utils.extended_scenario import RandomScenario

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import importlib.resources

# Path to patient parameter file
patient_file_path = importlib.resources.files('environments.simglucose.simglucose.params').joinpath('vpatient_params.csv')
PATIENT_PARA_FILE = str(patient_file_path)


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, args=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.args = args
        self.INSULIN_PUMP_HARDWARE = args.pump
        self.SENSOR_HARDWARE = args.sensor

        if patient_name is None:
            patient_name = 'adolescent#001'
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

        self.meal_announce_time = args.t_meal

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)

        # Announcement of meals and physical activities
        future_carb, remaining_time, day_hour, day_min, meal_type = self.announce_meal(meal_announce=None)
        future_activity, remaining_activity_time, activity_status = self.announce_activity()

        if self.reward_fun is None:
            state, reward, done, info = self.env.step(act)
            return state, reward, done, info
        else:
            state, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)

            # Updating 'info' dictionary with meal-related data
            info['future_carb'] = future_carb
            info['remaining_time'] = remaining_time
            info['day_hour'] = day_hour
            info['day_min'] = day_min
            info['meal_type'] = meal_type

            # Updating 'info' dictionary with activity-related data
            info['future_activity'] = future_activity  # (intensity, duration)
            info['remaining_activity_time'] = remaining_activity_time
            info['activity_intensity'] = future_activity[0] if future_activity else 0
            info['activity_duration'] = future_activity[1] if future_activity else 0
            info['activity_status'] = activity_status  # "Physical Activity Ongoing" or "No Activity"

        return state, reward, done, info

    def announce_meal(self, meal_announce=None):
        t = self.env.time.hour * 60 + self.env.time.minute  # Current time in minutes
        sampling_rate = self.sampling_time
        meal_type = 0

        for i, m_t in enumerate(self.env.scenario.scenario['meal']['time']):
            # Round up to sampling rate floor
            m_tr = m_t - (m_t % sampling_rate) if m_t % sampling_rate != 0 else m_t
            ma = self.meal_announce_time if meal_announce is None else meal_announce

            # Meal announcement
            if (m_tr - ma) <= t <= m_tr:
                meal_type = 0.3 if self.env.scenario.scenario['meal']['amount'][i] <= 40 else 1
                return (
                    self.env.scenario.scenario['meal']['amount'][i],
                    (m_tr - t),
                    self.env.time.hour,
                    self.env.time.minute,
                    meal_type
                )
            elif t < (m_tr - ma):  # Exit if no future meal is in this range
                break

        return 0, 0, self.env.time.hour, self.env.time.minute, meal_type

    def announce_activity(self):
        """
        Check if an activity is ongoing at the current time `t` based on the
        `activity` dictionary from the scenario.
        """
        t = self.env.time.hour * 60 + self.env.time.minute  # Current time in minutes
        sampling_rate = self.sampling_time  # Sampling rate from the sensor

        for i, a_t in enumerate(self.env.scenario.scenario['activity']['time']):
            # Define activity time range
            activity_start = a_t
            activity_end = a_t + self.env.scenario.scenario['activity']['duration'][i]

            if activity_start <= t <= activity_end:
                activity_intensity = self.env.scenario.scenario['activity']['intensity'][i]
                remaining_time = activity_end - t
                return (
                    (activity_intensity, self.env.scenario.scenario['activity']['duration'][i]),
                    remaining_time,
                    "Physical Activity Ongoing"
                )

            elif t < activity_start - sampling_rate:
                break

        # If no activity is ongoing
        return None, 0, "No Activity"

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash(self.np_random.integers(0, 1000)) % 2**31
        seed3 = hash(seed2 + 1) % 2 ** 31
        seed4 = hash(seed3 + 1) % 2 ** 31

        #hour = self.np_random.randint(low=0.0, high=24.0)
        hour = 23  #always start at midnight

        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        self.sampling_time = sensor.sample_time
        scenario = RandomScenario(start_time=start_time, seed=seed3, opt=self.args)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))