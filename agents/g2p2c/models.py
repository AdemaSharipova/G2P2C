import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import core
from utils.reward_func import composite_reward


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.n_features = args.n_features
        self.n_handcrafted_features = args.n_handcrafted_features  # Number of extra handcrafted features
        self.use_handcraft = args.use_handcraft  # Whether to use handcrafted features
        self.n_hidden = args.n_hidden  # Hidden size of LSTM
        self.n_layers = args.n_rnn_layers  # Number of LSTM layers
        self.bidirectional = args.bidirectional  # Bidirectional LSTM (True/False)
        self.directions = args.rnn_directions  # Number of LSTM directions
        self.LSTM = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )  # (seq_len, batch, input_size)

    def forward(self, s, feat, mode, activity_feat=None):
        if mode == "batch":
            output, (hid, cell) = self.LSTM(s)
            lstm_output = hid.view(hid.size(1), -1)  # ==> batch, layers * hidden size
            feat = feat.squeeze(1)
        else:
            s = s.unsqueeze(0)  # Add batch dimension for single forward pass
            output, (hid, cell) = self.LSTM(s)  # hid = layers * directions, batch, hidden
            lstm_output = hid.squeeze(1)  # Remove batch dimension
            lstm_output = torch.flatten(lstm_output)  # ==> layers * hidden size

        # Update for physical activities: Concatenate activity features, if provided
        if activity_feat is not None:
            if mode == "batch":
                feat = torch.cat((feat, activity_feat), dim=1)  # Concatenate along feature axis (batch mode)
            else:
                feat = torch.cat((feat, activity_feat), dim=0)  # Concatenate along feature axis (single mode)

        # Concatenate LSTM output and additional features
        if self.use_handcraft == 1:
            if mode == "batch":
                extract_states = torch.cat((lstm_output, feat), dim=1)  # ==> Output: torch.size[batch, 256 + feat]
            else:
                extract_states = torch.cat((lstm_output, feat), dim=0)  # ==> Output: torch.size[seq_len, 256 + feat]
        else:
            extract_states = lstm_output  # Use only LSTM output if handcrafted features are not enabled
        return extract_states, lstm_output

class GlucoseModel(nn.Module):
    def __init__(self, args, device):
        super(GlucoseModel, self).__init__()
        self.n_features = args.n_features
        self.device = device
        self.n_handcrafted_features = args.n_handcrafted_features  # Number of handcrafted features
        self.use_handcraft = args.use_handcraft  # Whether to use handcrafted features
        self.output = args.n_action  # Output dimension (e.g., continuous glucose prediction)
        self.n_hidden = args.n_hidden  # Hidden size of LSTM
        self.n_layers = args.n_rnn_layers  # Number of LSTM layers
        self.bidirectional = args.bidirectional  # Bidirectional LSTM (True/False)
        self.directions = args.rnn_directions  # Number of directions in LSTM (1 or 2)

        # Updated to dynamically adjust for input size
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor  # Final hidden layer

        # Fully connected layers
        self.fc_layer1 = nn.Linear(self.feature_extractor + self.output, self.last_hidden)

        # Output layers for glucose prediction
        self.cgm_mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.cgm_sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)

    def forward(self, extract_state, action, mode, activity_feat=None):
        concat_dim = 1 if (mode == 'batch') else 0

        # Include physical activity features if provided
        if activity_feat is not None:
            extract_state = torch.cat((extract_state, activity_feat), dim=concat_dim)

        # Concatenate extract_state with actions
        concat_state_action = torch.cat((extract_state, action), dim=concat_dim)

        # Pass through fully connected layers
        fc_output1 = F.relu(self.fc_layer1(concat_state_action))
        fc_output = fc_output1

        # Predict glucose mean (cgm_mu) and standard deviation (cgm_sigma)
        cgm_mu = F.tanh(self.cgm_mu(fc_output))
        cgm_sigma = F.softplus(self.cgm_sigma(fc_output) + 1e-5)

        # Sample from Normal distribution
        z = self.normal.sample()
        cgm = cgm_mu + cgm_sigma * z

        # Clamp output to ensure predictions remain in realistic bounds
        cgm = torch.clamp(cgm, -1, 1)

        return cgm_mu, cgm_sigma, cgm

class ValueModule(nn.Module):
    def __init__(self, args, device):
        super(ValueModule, self).__init__()
        self.device = device
        self.output = args.n_action  # Output dimension (e.g., value for each action)
        self.n_handcrafted_features = args.n_handcrafted_features  # Number of handcrafted features
        self.use_handcraft = args.use_handcraft  # Whether to use handcrafted features
        self.n_hidden = args.n_hidden  # Hidden size of LSTM
        self.n_layers = args.n_rnn_layers  # Number of LSTM layers
        self.directions = args.rnn_directions  # Number of LSTM directions (1 for unidirectional, 2 for bidirectional)

        # Feature extractor size dynamically adjusts for physical activity features
        self.feature_extractor = (
                self.n_hidden * self.n_layers * self.directions
                + (self.n_handcrafted_features * self.use_handcraft)
        )
        self.last_hidden = self.feature_extractor * 2

        # Fully connected layers for value estimation
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)

        # Output layer for value estimation
        self.value = NormedLinear(self.last_hidden, self.output, scale=0.1)

    def forward(self, extract_states, activity_feat=None):
        if activity_feat is not None:
            concat_dim = 1  # Concatenate along feature axis
            extract_states = torch.cat((extract_states, activity_feat), dim=concat_dim)

        # Pass through the network
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))

        # Output estimated value
        value = self.value(fc_output)
        return value

class ActorNetwork(nn.Module):
    def __init__(self, args, device):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.args = args

        # Submodules
        self.FeatureExtractor = FeatureExtractor(args)
        self.GlucoseModel = GlucoseModel(args, self.device)
        self.ActionModule = ActionModule(args, self.device)

        self.distribution = torch.distributions.Normal
        self.planning_n_step = args.planning_n_step
        self.n_planning_simulations = args.n_planning_simulations

        # Target variables (normalized for the model)
        self.glucose_target = core.linear_scaling(
            x=112.5, x_min=self.args.glucose_min, x_max=self.args.glucose_max
        )
        self.t_to_meal = core.linear_scaling(
            x=0, x_min=0, x_max=self.args.t_meal
        )  # Time to the next meal

    def forward(self, s, feat, old_action, mode, is_training=False, activity_feat=None):  # Updated for activity_feat
        # Updated for activity_feat
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode, activity_feat=activity_feat)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)

        if mode == 'forward':
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(
                lstmOut, action.detach(), mode, activity_feat=activity_feat  # Passing activity_feat
            )
        else:
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(
                lstmOut, old_action.detach(), mode, activity_feat=activity_feat  # Passing activity_feat
            )

        return mu, sigma, action, log_prob, cgm_mu, cgm_sigma, cgm

    def update_state(self, s, cgm_pred, action, batch_size):
        if batch_size == 1:
            if self.args.n_features == 2:
                s_new = torch.cat((cgm_pred, action), dim=0)
            if self.args.n_features == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(1, device=self.device)), dim=0)
            s_new = s_new.unsqueeze(0)
            s = torch.cat((s[1:self.args.feature_history, :], s_new), dim=0)
        else:
            if self.args.n_features == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(batch_size, 1, device=self.device)),
                                  dim=1)
            if self.args.n_features == 2:
                s_new = torch.cat((cgm_pred, action), dim=1)
            s_new = s_new.unsqueeze(1)
            s = torch.cat((s[:, 1:self.args.feature_history, :], s_new), dim=1)
        return s

    def expert_search(self, s, feat, rew_norm_var, mode, activity_feat=None):  # Updated for activity_feat
        pi, mu, sigma, s_e, f_e, r = self.expert_MCTS_rollout(s, feat, mode, rew_norm_var, activity_feat=activity_feat)
        return pi, mu, sigma, s_e, f_e, r

    def expert_MCTS_rollout(self, s, feat, mode, rew_norm_var=1, activity_feat=None):  # Updated for activity_feat
        batch_size = s.shape[0]
        first_action, first_mu, first_sigma, cum_reward, mu, sigma = 0, 0, 0, 0, 0, 0
        for i in range(self.planning_n_step):
            extract_states, lstmOut = self.FeatureExtractor.forward(
                s, feat, mode, activity_feat=activity_feat  # Passing activity_feat
            )
            extract_states, lstmOut = extract_states.detach(), lstmOut.detach()
            mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
            if i == 0:
                first_action = action
                first_mu = mu
                first_sigma = sigma
            _, _, cgm_pred = self.GlucoseModel.forward(lstmOut, action, mode, activity_feat=activity_feat)
            bg = core.inverse_linear_scaling(y=cgm_pred.detach().cpu().numpy(), x_min=self.args.glucose_min, x_max=self.args.glucose_max)
            reward = np.array([[composite_reward(self.args, state=xi, reward=None)] for xi in bg])
            reward = reward / (math.sqrt(rew_norm_var + 1e-8))
            reward = np.clip(reward, 10, 10)
            discount = (self.args.gamma ** i)
            cum_reward += (reward * discount)

            action = action.detach()
            cgm_pred = cgm_pred.detach()
            pump_action = self.args.action_scale * (torch.exp((action - 1) * 4))
            action = core.linear_scaling(x=pump_action, x_min=self.args.insulin_min, x_max=self.args.insulin_max)

            s = self.update_state(s, cgm_pred, action, batch_size)
            feat[0] += 1  # Quick fix for integrating features
        cum_reward = torch.as_tensor(cum_reward, dtype=torch.float32, device=self.device)
        return first_action, first_mu, first_sigma, s, feat, cum_reward

    def horizon_error(self, s, feat, actions, real_glucose, mode, activity_feat=None):
        horizon_error = 0
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        for i in range(0, len(actions)):
            cur_action = torch.as_tensor(actions[i], dtype=torch.float32, device=self.device).reshape(1)
            extract_states, lstmOut = self.FeatureExtractor.forward(
                s, feat, mode, activity_feat=activity_feat  # Passing activity_feat
            )
            extract_states, lstmOut = extract_states.detach(), lstmOut.detach()

            cgm_mu, cgm_sigma, cgm_pred = self.GlucoseModel.forward(lstmOut, cur_action, mode,
                                                                    activity_feat=activity_feat)
            pred = core.inverse_linear_scaling(
                y=cgm_pred.detach().cpu().numpy(), x_min=self.args.glucose_min, x_max=self.args.glucose_max
            )
            horizon_error += ((pred - real_glucose[i]) ** 2)
            s = self.update_state(s, cgm_pred, cur_action, batch_size=1)
        return horizon_error / len(actions)


class CriticNetwork(nn.Module):
    def __init__(self, args, device):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = FeatureExtractor(args)
        self.ValueModule = ValueModule(args, device)
        self.aux_mode = args.aux_mode
        self.GlucoseModel = GlucoseModel(args, device)

    def forward(self, s, feat, action, cgm_pred=True, mode='forward', activity_feat=None):  # Updated for activity_feat
        # Extract states and features
        extract_states, lstmOut = self.FeatureExtractor.forward(s, feat, mode, activity_feat=activity_feat)
        value = self.ValueModule.forward(extract_states)
        cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(
            lstmOut, action.detach(), mode, activity_feat=activity_feat  # Pass activity_feat
        ) if cgm_pred else (None, None, None)
        return value, cgm_mu, cgm_sigma, cgm


class ActorCritic(nn.Module):
    def __init__(self, args, load, actor_path, critic_path, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.experiment_dir = args.experiment_dir
        self.Actor = ActorNetwork(args, device)
        self.Critic = CriticNetwork(args, device)

        if load:
            self.Actor = torch.load(actor_path, map_location=device)
            self.Critic = torch.load(critic_path, map_location=device)

        self.distribution = torch.distributions.Normal
        self.is_testing_worker = False

    def predict(self, s, feat, activity_feat=None):  # Updated for activity_feat
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        mean, std, action, log_prob, a_cgm_mu, a_cgm_sigma, a_cgm = self.Actor(
            s, feat, None, mode='forward', is_training=self.is_testing_worker, activity_feat=activity_feat
        )
        state_value, c_cgm_mu, c_cgm_sigma, c_cgm = self.Critic(
            s, feat, action, cgm_pred=True, mode='forward', activity_feat=activity_feat
        )
        return (
            (mean, std, action, log_prob, a_cgm_mu, a_cgm_sigma, a_cgm),
            (state_value, c_cgm_mu, c_cgm_sigma, c_cgm),
        )

    def get_action(self, s, feat, activity_feat=None):  # Updated for activity_feat
        (mu, std, act, log_prob, a_cgm_mu, a_cgm_sig, a_cgm), (s_val, c_cgm_mu, c_cgm_sig, c_cgm) = self.predict(
            s, feat, activity_feat=activity_feat
        )
        data = dict(
            mu=mu,
            std=std,
            action=act,
            log_prob=log_prob,
            state_value=s_val,
            a_cgm_mu=a_cgm_mu,
            a_cgm_sigma=a_cgm_sig,
            c_cgm_mu=c_cgm_mu,
            c_cgm_sigma=c_cgm_sig,
            a_cgm=a_cgm,
            c_cgm=c_cgm,
        )
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def get_final_value(self, s, feat, activity_feat=None):  # Updated for activity_feat
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        state_value, _, _, _ = self.Critic(
            s, feat, action=None, cgm_pred=False, mode='forward', activity_feat=activity_feat
        )
        return state_value.detach().cpu().numpy()

    def evaluate_actor(self, state, action, feat, activity_feat=None):  # Updated for activity_feat
        action_mean, action_std, _, _, a_cgm_mu, a_cgm_sigma, _ = self.Actor(
            state, feat, action, mode='batch', activity_feat=activity_feat
        )
        dist = self.distribution(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy, a_cgm_mu, a_cgm_sigma

    def evaluate_critic(self, state, feat, action, cgm_pred, activity_feat=None):  # Updated for activity_feat
        state_value, c_cgm_mu, c_cgm_sigma, _ = self.Critic(
            state, feat, action, cgm_pred=cgm_pred, mode='batch', activity_feat=activity_feat
        )
        return torch.squeeze(state_value), c_cgm_mu, c_cgm_sigma

    def save(self, episode):
        actor_path = self.experiment_dir + f'/checkpoints/episode_{episode}_Actor.pth'
        critic_path = self.experiment_dir + f'/checkpoints/episode_{episode}_Critic.pth'
        torch.save(self.Actor, actor_path)
        torch.save(self.Critic, critic_path)


class ActionModule(nn.Module):
    def __init__(self, args, device):
        super(ActionModule, self).__init__()
        self.device = device
        self.args = args
        self.n_handcrafted_features = args.n_handcrafted_features
        self.use_handcraft = args.use_handcraft
        self.output = args.n_action
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_rnn_layers
        self.directions = args.rnn_directions

        # Input size of the feature extractor dynamically considers handcrafted features
        self.feature_extractor = (
                self.n_hidden * self.n_layers * self.directions
                + (self.n_handcrafted_features * self.use_handcraft)
        )
        self.last_hidden = self.feature_extractor * 2

        # Fully connected layers
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states, action_type='N'):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))

        # Compute mean and sigma for the action distribution
        mu = torch.tanh(self.mu(fc_output))
        sigma = torch.sigmoid(self.sigma(fc_output)) + 1e-5

        # Sample action from normal distribution
        z = self.normalDistribution(0, 1).sample()
        action = mu + sigma * z
        action = torch.clamp(action, -1, 1)

        try:
            dst = self.normalDistribution(mu, sigma)
            log_prob = dst.log_prob(action[0])
        except ValueError:
            log_prob = torch.ones(2, 1, device=self.device) * self.args.glucose_target

        return mu, sigma, action, log_prob


def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out
