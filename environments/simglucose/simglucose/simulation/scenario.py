import logging
from collections import namedtuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
Action = namedtuple('scenario_action', ['meal', 'physical_activity'])


class Scenario:
    def __init__(self, start_time=None):
        if start_time is None:
            now = datetime.now()
            start_hour = timedelta(hours=float(
                input('Input simulation start time (hr): ')))
            start_time = datetime.combine(now.date(),
                                          datetime.min.time()) + start_hour
            print('Simulation start time is set to {}.'.format(start_time))
        self.start_time = start_time

    def get_action(self, t):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class CustomScenario(Scenario):
    def __init__(self, start_time=None, scenario=None):

        Scenario.__init__(self, start_time=start_time)
        if scenario is None:
            scenario = self.input_scenario()
        self.scenario = scenario

    def get_action(self, t):
        times, actions = tuple(zip(*self.scenario))
        times2compare = [parseTime(time, self.start_time) for time in times]
        if t in times2compare:
            idx = times2compare.index(t)
            return Action(meal=actions[idx][0], physical_activity=actions[idx][1])
        else:
            return Action(meal=0, physical_activity=0)

    def reset(self):
        pass

    @staticmethod
    def input_scenario():
        scenario = []

        print('Input a custom scenario ...')

        # Input Meals
        breakfast_time = float(input('Input breakfast time (hr): '))
        breakfast_size = float(input('Input breakfast size (g): '))
        scenario.append((breakfast_time, (breakfast_size, 0)))

        lunch_time = float(input('Input lunch time (hr): '))
        lunch_size = float(input('Input lunch size (g): '))
        scenario.append((lunch_time, (lunch_size, 0)))

        dinner_time = float(input('Input dinner time (hr): '))
        dinner_size = float(input('Input dinner size (g): '))
        scenario.append((dinner_time, (dinner_size, 0)))

        while True:
            snack_time = float(input('Input snack time (hr): '))
            snack_size = float(input('Input snack size (g): '))
            scenario.append((snack_time, (snack_size, 0)))

            go_on = input('Continue input snack (y/n)? ')
            if go_on == 'n':
                break

        # Input Physical Activities
        print('Input physical activities ...')

        while True:
            pa_time = float(input('Input physical activity time (hr): '))
            pa_intensity = float(input('Input physical activity intensity: '))
            scenario.append((pa_time, (0, pa_intensity)))

            go_on = input('Continue input physical activity (y/n)? ')
            if go_on == 'n':
                break

        return scenario


def parseTime(time, start_time):
    if isinstance(time, (int, float)):
        t = start_time + timedelta(minutes=round(time * 60.0))
    elif isinstance(time, timedelta):
        t_sec = time.total_seconds()
        t_min = round(t_sec / 60.0)
        t = start_time + timedelta(minutes=t_min)
    elif isinstance(time, datetime):
        t = time
    else:
        raise ValueError('Expect time to be int, float, timedelta, datetime')
    return t


if __name__ == '__main__':
    from datetime import time
    import copy

    now = datetime.now()
    t0 = datetime.combine(now.date(), time(6, 0, 0, 0))
    t = copy.deepcopy(t0)
    sim_time = timedelta(days=2)

    # Example Scenario with User Input
    scenario = CustomScenario()
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
