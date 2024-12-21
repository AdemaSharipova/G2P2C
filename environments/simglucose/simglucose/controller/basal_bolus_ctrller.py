from scipy.constants import physical_constants

from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import importlib.resources
import logging

logger = logging.getLogger(__name__)

patient_file_path = importlib.resources.files('simglucose.params').joinpath('vpatient_params.csv')
PATIENT_PARA_FILE = str(patient_file_path)

quest_file_path = importlib.resources.files('simglucose.params').joinpath('Quest.csv')
CONTROL_QUEST = str(quest_file_path)


class BBController(Controller):
    """
    This is a Basal-Bolus Controller that is typically practiced by a Type-1
    Diabetes patient. The performance of this controller can serve as a
    baseline when developing a more advanced controller.
    """
    def __init__(self, target=140):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)
        self.target = target

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')
        physical_activity = kwargs.get('physical_activity', 0)

        action = self._bb_policy(
            pname,
            meal,
            observation.CGM,
            sample_time,
            physical_activity=physical_activity,
        )
        return action

    def _bb_policy(self, name, meal, glucose, env_sample_time, physical_activity=0):
        """
        Helper function to compute the basal and bolus amount.

        The basal insulin is based on the insulin amount to keep the blood
        glucose in the steady state when there is no (meal) disturbance.
               basal = u2ss (pmol/(L*kg)) * body_weight (kg) / 6000 (U/min)

        The bolus amount is computed based on the current glucose level, the
        target glucose level, the patient's correction factor and the patient's
        carbohydrate ratio.
               bolus = ((carbohydrate / carbohydrate_ratio) +
                       (current_glucose - target_glucose) / correction_factor)
                       / sample_time
        NOTE the bolus computed from the above formula is in unit U. The
        simulator only accepts insulin rate. Hence the bolus is converted to
        insulin rate.
        """
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        basal = u2ss * BW / 6000  # unit: U/min
        
        # Adjust basal insulin based on physical activity level
        if meal > 0:
            logger.info('Calculating bolus ...')
            logger.debug('glucose = {}'.format(glucose))
            bolus = (meal / quest.CR.values + (glucose > 150)
                     * (glucose - self.target) / quest.CF.values).item()  # unit: U
        else:
            bolus = 0  # unit: U

        # This is to convert bolus in total amount (U) to insulin rate (U/min).
        # The simulation environment does not treat basal and bolus
        # differently. The unit of Action.basal and Action.bolus are the same
        # (U/min).
        activity_factor = 0.3  # This is a hypothetical factor, adjust as needed
        bolus *= max(0.5, 1 - physical_activity * activity_factor)

        bolus = bolus / env_sample_time  # unit: U/min
        return Action(basal=basal, bolus=bolus)

    def reset(self):
        pass
