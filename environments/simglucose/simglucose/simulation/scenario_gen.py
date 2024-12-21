from simglucose.simulation.scenario import Action, Scenario
import numpy as np
from scipy.stats import truncnorm
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RandomScenario(Scenario):
    def __init__(self, start_time=None, seed=None):
        Scenario.__init__(self, start_time=start_time)
        self.seed = seed

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        # if t_sec < 1:
        #     logger.info('Creating new one day scenario ...')
        #     self.scenario = self.create_scenario()

        t_min = np.floor(t_sec / 60.0)


        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            meal_amt = self.scenario['meal']['amount'][idx]
        else:
            meal_amt = 0

        pa_intensity = 0
        for start_time, intensity, duration in zip(
                self.scenario['activity']['time'],
                self.scenario['activity']['intensity'],
                self.scenario['activity']['duration']):
            if start_time <= t_min < start_time + duration:
                logger.info('Time for physical activity!')
                pa_intensity = intensity  # Activity is ongoing
                break

        return Action(meal=meal_amt, physical_activity=pa_intensity)

    def create_scenario(self):
        scenario = {'meal': {'time': [], 'amount': []}, 'activity': {'time': [], 'intensity': [], 'duration': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60 # lower bounds for meal time
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60 # upper bounds for meal time
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60 # mean meal times
        time_sigma = np.array([60, 30, 60, 30, 60, 30]) # standard deviation
        amount_mu = [45, 10, 70, 10, 80, 10]
        amount_sigma = [10, 5, 10, 5, 10, 5]

        # Probability of physical activity
        prob_activity = [0.7, 0.2, 0.5]  # probabilities for morning, afternoon, and evening activities

        # Timing for physical activity
        time_lb_activity = np.array([6, 12, 18]) * 60  # lower bounds for physical activity time in minutes
        time_ub_activity = np.array([9, 14, 21]) * 60  # upper bounds for physical activity time in minutes
        time_mu_activity = np.array([7.5, 13, 19.5]) * 60  # mean physical activity times in minutes
        time_sigma_activity = np.array([90, 120, 90])  # standard deviation of physical activity times in minutes

        # Intensity for physical activity
        intensity_mu_activity = [0.6, 0.4, 0.5]  # normalized intensities
        intensity_mu_activity = [mu * 80 + 40 for mu in intensity_mu_activity]

        intensity_sigma_activity = [0.15 * 50, 0.1 * 50,
                                    0.15 * 50]  # standard deviations

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma,
                amount_mu, amount_sigma):
            if self.random_gen.rand() < p:
                tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                               b=(tub - tbar) / tsd,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=self.random_gen))
                scenario['meal']['time'].append(tmeal)
                scenario['meal']['amount'].append(
                    max(round(self.random_gen.normal(mbar, msd)), 0))


        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob_activity, time_lb_activity, time_ub_activity, time_mu_activity, time_sigma_activity,
                intensity_mu_activity, intensity_sigma_activity):
            if self.random_gen.rand() < p:
                # Time
                tact = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                              b=(tub - tbar) / tsd,
                                              loc=tbar,
                                              scale=tsd,
                                              random_state=self.random_gen))
                # Intensity
                intensity = max(round(self.random_gen.normal(mbar, msd), 2), 0)

                # Duration (example: random duration between 30 to 90 minutes)
                duration = self.random_gen.randint(30, 90)

                scenario['activity']['time'].append(tact)
                scenario['activity']['intensity'].append(intensity)
                scenario['activity']['duration'].append(duration)

        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()


if __name__ == '__main__':
    from datetime import time
    from datetime import timedelta
    import copy
    now = datetime.now()
    t0 = datetime.combine(now.date(), time(6, 0, 0, 0))
    t = copy.deepcopy(t0)
    sim_time = timedelta(days=2)

    scenario = RandomScenario(seed=1)
    meals = []
    activities = []
    T = []
    while t < t0 + sim_time:
        action = scenario.get_action(t)
        meals.append(action.meal)
        activities.append(action.physical_activity)
        T.append(t)
        t += timedelta(minutes=1)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(T, meals, label='Meals')
    ax1.set_ylabel('Meal (g CHO)')
    ax1.legend()

    ax2.plot(T, activities, label='Physical Activity')
    ax2.set_ylabel('PA Intensity')
    ax2.legend()

    ax = plt.gca()
    ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    plt.show()
