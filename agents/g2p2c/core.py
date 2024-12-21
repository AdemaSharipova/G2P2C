import torch
import numpy as np
from utils import core


class CGPredHorizon:
    def __init__(self, args):
        self.horizon = args.planning_n_step
        self.real_glucose = np.zeros(self.horizon, dtype=np.float32)
        self.actions = np.zeros(self.horizon, dtype=np.float32)
        self.cur_state = 0
        self.feat = 0
        self.activity_feat = np.zeros((self.horizon, args.n_activity_features), dtype=np.float32)  # Added
        self.counter = 0

    def update(self, cur_state, feat, activity_feat, action, real_glucose, policy):
        done, err = False, 0
        if self.counter == 0:
            self.cur_state = cur_state
            self.feat = feat
        self.activity_feat[self.counter] = activity_feat  # Store activity_feat
        self.actions[self.counter] = action
        self.real_glucose[self.counter] = real_glucose
        self.counter += 1
        if self.counter == self.horizon:
            done = True
            err = policy.Actor.horizon_error(
                self.cur_state, self.feat, self.activity_feat, self.actions, self.real_glucose, mode='forward'
            )
            self.actions = np.zeros(self.horizon, dtype=np.float32)
            self.real_glucose = np.zeros(self.horizon, dtype=np.float32)
            self.activity_feat = np.zeros((self.horizon, policy.args.n_activity_features), dtype=np.float32)
            self.counter = 0
        return done, err

    def reset(self):
        self.actions = np.zeros(self.horizon, dtype=np.float32)
        self.real_glucose = np.zeros(self.horizon, dtype=np.float32)
        self.activity_feat = np.zeros((self.horizon, self.activity_feat.shape[1]), dtype=np.float32)  # Reset
        self.counter = 0


class BGPredBuffer:
    def __init__(self, args):
        self.n_bgp_steps = args.n_bgp_steps
        self.args = args
        self.actor_predictions = []
        self.critic_predictions = []
        self.real_glucose = []
        self.activity_feat = []  # Added to store physical activity features

    def clear(self):
        self.actor_predictions = []
        self.critic_predictions = []
        self.real_glucose = []
        self.activity_feat = []  # Clear activity features

    def update(self, act_pred, critic_pred, real, activity_feat):
        act_pred = act_pred.flatten()
        critic_pred = critic_pred.flatten()
        for i in range(0, len(act_pred)):
            act_pred[i] = core.inverse_linear_scaling(y=act_pred[i], x_min=self.args.glucose_min,
                                                      x_max=self.args.glucose_max)
            critic_pred[i] = core.inverse_linear_scaling(y=critic_pred[i], x_min=self.args.glucose_min,
                                                         x_max=self.args.glucose_max)

        self.actor_predictions.append(act_pred)
        self.critic_predictions.append(critic_pred)
        self.real_glucose.append(real)
        self.activity_feat.append(activity_feat)  # Append activity features

    def calc_accuracy(self):
        a_rmse_error, c_rmse_error, count = 0, 0, 0
        if len(self.real_glucose) > self.n_bgp_steps:
            for i in range(0, len(self.real_glucose) - self.n_bgp_steps):
                count += 1
                a_rmse_error += np.sum(np.square(self.actor_predictions[i] - self.real_glucose[i:i + self.n_bgp_steps]))
                c_rmse_error += np.sum(
                    np.square(self.critic_predictions[i] - self.real_glucose[i:i + self.n_bgp_steps]))
        else:
            a_rmse_error, c_rmse_error, count = 0, 0, 1
        return np.sqrt(a_rmse_error / (self.n_bgp_steps * count)), np.sqrt(c_rmse_error / (self.n_bgp_steps * count))



class AuxiliaryBuffer:
    def __init__(self, args, device):
        self.size = args.aux_buffer_max
        self.n_bgp_steps = args.n_bgp_steps
        self.bgp_pred_mode = args.bgp_pred_mode
        self.n_activity_features = args.n_activity_features  # Number of activity features
        self.old_states = torch.zeros(self.size, args.feature_history, args.n_features, device=device,
                                      dtype=torch.float32)
        self.handcraft_feat = torch.zeros(self.size, 1, args.n_handcrafted_features, device=device, dtype=torch.float32)
        self.activity_feat = torch.zeros(self.size, 1, self.n_activity_features, device=device,
                                         dtype=torch.float32)  # Added
        self.actions = torch.zeros(self.size, 1, device=device, dtype=torch.float32)
        self.logprob = torch.zeros(self.size, 1, device=device, dtype=torch.float32)
        self.value_target = torch.zeros(self.size, device=device, dtype=torch.float32)
        self.aux_batch_size = args.aux_batch_size
        self.device = device
        self.buffer_filled = False
        self.buffer_level = 0

        if self.bgp_pred_mode:
            self.cgm_target = torch.zeros(self.size, 1, self.n_bgp_steps, device=device, dtype=torch.float32)
        else:
            self.cgm_target = torch.zeros(self.size, 1, device=device, dtype=torch.float32)

    def update(self, s, hand_feat, activity_feat, cgm_target, actions, first_flag):
        """Add activity_feat to the update."""
        if self.bgp_pred_mode:
            s, hand_feat, activity_feat, actions, cgm_target = self.prepare_bgp_prediction(
                s, hand_feat, activity_feat, cgm_target, actions, first_flag
            )
        else:
            cgm_target = cgm_target.view(-1, 1)
            update_size = actions.shape[0]
            self.old_states = torch.cat((self.old_states[update_size:, :, :], s), dim=0)
            self.handcraft_feat = torch.cat((self.handcraft_feat[update_size:, :, :], hand_feat), dim=0)
            self.activity_feat = torch.cat((self.activity_feat[update_size:, :, :], activity_feat),
                                           dim=0)  # Update activity_feat
            self.cgm_target = torch.cat((self.cgm_target[update_size:, :], cgm_target), dim=0)
            self.actions = torch.cat((self.actions[update_size:], actions), dim=0)

        if not self.buffer_filled:
            self.buffer_level += update_size
            if self.buffer_level >= self.size:
                self.buffer_filled = True

    def prepare_bgp_prediction(self, s_hist, s_handcraft, activity_feat, cgm_target, act, first_flag):
        """Update prepare_bgp_prediction to include activity_feat."""
        buffer_len = s_hist.shape[0]
        bgp_first_flag = first_flag.view(-1).cpu().numpy()
        bgp_cgm_target = cgm_target.cpu().numpy()
        bgp_s_hist = s_hist.cpu().numpy()
        bgp_s_handcraft = s_handcraft.cpu().numpy()
        bgp_activity_feat = activity_feat.cpu().numpy()  # Added activity_feat
        bgp_act = act.cpu().numpy()
        new_cgm_target = np.zeros(core.combined_shape(buffer_len, (1, self.n_bgp_steps)), dtype=np.float32)
        delete_arr = list(range((buffer_len - self.n_bgp_steps + 1), buffer_len))

        for ii in range(0, buffer_len - (self.n_bgp_steps - 1)):
            flag_status = np.sum(bgp_first_flag[ii + 1:ii + self.n_bgp_steps])
            new_cgm_target[ii] = bgp_cgm_target[ii:ii + self.n_bgp_steps]
            if flag_status >= 1:
                delete_arr.append(ii)

        bgp_s_hist = torch.from_numpy(np.delete(bgp_s_hist, delete_arr, axis=0)).to(self.device)
        bgp_s_handcraft = torch.from_numpy(np.delete(bgp_s_handcraft, delete_arr, axis=0)).to(self.device)
        bgp_activity_feat = torch.from_numpy(np.delete(bgp_activity_feat, delete_arr, axis=0)).to(
            self.device)  # Updated
        bgp_act = torch.from_numpy(np.delete(bgp_act, delete_arr, axis=0)).to(self.device)
        new_cgm_target = torch.from_numpy(np.delete(new_cgm_target, delete_arr, axis=0)).to(self.device)
        return bgp_s_hist, bgp_s_handcraft, bgp_activity_feat, bgp_act, new_cgm_target



class Memory:
    def __init__(self, args, device):
        self.size = args.n_step
        self.device = device
        self.feature_hist = args.feature_history
        self.features = args.n_features
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.handcrafted_features = args.n_handcrafted_features
        self.observation = np.zeros(core.combined_shape(self.size, (self.feature_hist, self.features)), dtype=np.float32)
        self.state_features = np.zeros(core.combined_shape(self.size, (1, self.handcrafted_features)), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.state_values = np.zeros(self.size + 1, dtype=np.float32)
        self.logprobs = np.zeros(self.size, dtype=np.float32)
        self.first_flag = np.zeros(self.size + 1, dtype=np.bool_)
        self.cgm_target = np.zeros(self.size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size
        self.activity_feat = np.zeros(core.combined_shape(self.size, (1, args.n_activity_features)), dtype=np.float32)

    def store(self, obs, features, act, rew, val, logp, cgm_target, counter, activity_feat):
        assert self.ptr < self.max_size
        self.observation[self.ptr] = obs
        self.state_features[self.ptr] = features
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.state_values[self.ptr] = val
        self.logprobs[self.ptr] = logp
        self.first_flag[self.ptr] = True if counter == 0 else False
        self.cgm_target[self.ptr] = cgm_target
        self.ptr += 1
        self.activity_feat[self.ptr] = activity_feat  # Store activity_feat

    def finish_path(self, final_v):
        self.state_values[self.ptr] = final_v
        self.first_flag[self.ptr] = False

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(
            obs=self.observation, feat=self.state_features, activity_feat=self.activity_feat,  # Include activity_feat
            act=self.actions, v_pred=self.state_values, logp=self.logprobs,
            first_flag=self.first_flag, reward=self.rewards, cgm_target=self.cgm_target
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}