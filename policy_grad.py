import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet2Layer:
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=4):
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)

        self.W3 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b3 = np.zeros(output_dim)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def apply_dropout(self, activation, dropout_rate=0.5):
        mask = (np.random.rand(*activation.shape) > dropout_rate).astype(np.float32)
        return activation * mask / (1.0 - dropout_rate)  # scale during training

    def forward(self, state, train=True):
        self.state = state
        self.z1 = self.W1 @ state + self.b1
        self.a1 = self.leaky_relu(self.z1)
        if train:
            self.a1 = self.apply_dropout(self.a1, dropout_rate=0.1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.leaky_relu(self.z2)
        if train:
            self.a2 = self.apply_dropout(self.a2, dropout_rate=0.1)

        self.output = self.W3 @ self.a2 + self.b3

        self.mu = 1 / (1 + np.exp(-self.output[:2]))
        self.log_sigma = np.clip(self.output[2:], -3.0, -2.0)
        epsilon = 1e-8
        self.sigma = np.exp(self.log_sigma) + epsilon

        return self.mu, self.sigma

    def sample_action(self):
        action = np.random.normal(self.mu, self.sigma)
        return np.clip(action, 0.0, 1.0)

    def log_prob(self, action):
        var = self.sigma ** 2
        logp = -0.5 * (((action - self.mu)**2) / var + np.log(2 * np.pi * var))
        return np.sum(logp)

    def compute_gradients(self, action):
        raw_mu = self.output[:2]
        mu = self.mu
        sigma = self.sigma

        delta = (action - mu) / (sigma**2)
        dmu_doutput = mu * (1 - mu)  # derivative of sigmoid

        dlogp_dmu_raw = delta * dmu_doutput
        dlogp_dlogsigma = -1 + ((action - mu)**2) / (sigma**2)

        grad_output = np.concatenate([dlogp_dmu_raw, dlogp_dlogsigma])

        dW3 = np.outer(grad_output, self.a2)
        db3 = grad_output

        da2 = self.W3.T @ grad_output
        dz2 = da2 * self.leaky_relu_derivative(self.z2)

        dW2 = np.outer(dz2, self.a1)
        db2 = dz2

        da1 = self.W2.T @ dz2
        dz1 = da1 * self.leaky_relu_derivative(self.z1)

        dW1 = np.outer(dz1, self.state)
        db1 = dz1

        grads = (dW1, db1, dW2, db2, dW3, db3)
        return self.clip_grads(grads)

    def clip_grads(self, grads, max_norm=1.0):
        return tuple(np.clip(g, -max_norm, max_norm) for g in grads)

    def update(self, grads, lr=1e-3):
        dW1, db1, dW2, db2, dW3, db3 = grads
        self.W1 += lr * dW1
        self.b1 += lr * db1
        self.W2 += lr * dW2
        self.b2 += lr * db2
        self.W3 += lr * dW3
        self.b3 += lr * db3

    def save_weights(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load_weights(self, filename):
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']


class PolicyNet2LayerTorch(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        x = self.leaky_relu(self.fc1(state))
        x = self.leaky_relu(self.fc2(x))
        output = self.fc3(x)

        mu = torch.sigmoid(output[:2])
        log_sigma = torch.clamp(output[2:], min=-3.0, max=-2.0)
        sigma = torch.exp(log_sigma)

        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        action_clipped = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum()

        return action_clipped, log_prob, mu, sigma

    def sample_action(self, mu, sigma):
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        return torch.clamp(action, 0.0, 1.0), dist.log_prob(action).sum()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
