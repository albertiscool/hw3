"""
Rainbow DQN for Gridworld (Random Mode)
Combines 6 key improvements into one unified agent:
  1. Double DQN
  2. Dueling DQN
  3. Prioritized Experience Replay
  4. Multi-step Learning (n-step returns)
  5. Noisy Networks (for exploration)
  6. Distributional RL (Categorical DQN / C51)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math
from collections import deque
from Gridworld import Gridworld

# ============================================================
# 1. Noisy Linear Layer (replaces epsilon-greedy exploration)
# ============================================================
class NoisyLinear(nn.Module):
    """Factorised Gaussian Noise layer for exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ============================================================
# 2. Dueling + Noisy + Distributional (C51) Network
# ============================================================
class RainbowNetwork(nn.Module):
    """
    Network architecture combining:
      - Dueling streams (Value + Advantage)
      - Noisy layers (parameter-space exploration)
      - Distributional output (C51 atoms)
    """
    def __init__(self, input_dim=64, n_actions=4, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))

        # Shared feature layer
        self.fc1 = nn.Linear(input_dim, 128)

        # Value stream (Dueling)
        self.val_noisy1 = NoisyLinear(128, 128)
        self.val_noisy2 = NoisyLinear(128, n_atoms)

        # Advantage stream (Dueling)
        self.adv_noisy1 = NoisyLinear(128, 128)
        self.adv_noisy2 = NoisyLinear(128, n_actions * n_atoms)

    def forward(self, x):
        batch = x.size(0)
        feat = F.relu(self.fc1(x))

        # Dueling: Value stream
        v = F.relu(self.val_noisy1(feat))
        v = self.val_noisy2(v).view(batch, 1, self.n_atoms)  # (B, 1, atoms)

        # Dueling: Advantage stream
        a = F.relu(self.adv_noisy1(feat))
        a = self.adv_noisy2(a).view(batch, self.n_actions, self.n_atoms)  # (B, actions, atoms)

        # Combine (Dueling aggregation in log-space for numerical stability)
        q_atoms = v + a - a.mean(dim=1, keepdim=True)  # (B, actions, atoms)
        log_probs = F.log_softmax(q_atoms, dim=2)
        return log_probs

    def q_values(self, x):
        """Get expected Q values from the distribution."""
        log_probs = self.forward(x)
        probs = log_probs.exp()
        q = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        return q

    def reset_noise(self):
        self.val_noisy1.reset_noise()
        self.val_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()


# ============================================================
# 3. Prioritized Experience Replay (PER)
# ============================================================
class SumTree:
    """Binary sum tree for efficient priority sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay using SumTree."""
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5
        self.max_priority = 1.0

    def add(self, experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback: resample
                s = random.uniform(0, self.tree.total - 1e-5)
                idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        priorities = np.array(priorities, dtype=np.float32) + self.epsilon
        sampling_probs = priorities / self.tree.total
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ============================================================
# 4. N-step Return Buffer
# ============================================================
class NStepBuffer:
    """Accumulates n-step transitions before storing to main buffer."""
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def add(self, transition):
        self.buffer.append(transition)

    def get(self):
        """Compute n-step return and return the compressed transition."""
        state, action, _, _, _ = self.buffer[0]
        _, _, _, next_state, done = self.buffer[-1]
        reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            reward += (self.gamma ** i) * r
            if d:
                next_state = self.buffer[i][-2]
                done = True
                break
        return (state, action, reward, next_state, done)

    def is_ready(self):
        return len(self.buffer) == self.n_step

    def reset(self):
        self.buffer.clear()


# ============================================================
# 5. Rainbow Agent (ties everything together)
# ============================================================
class RainbowAgent:
    def __init__(self, n_atoms=51, v_min=-10, v_max=10, gamma=0.99,
                 n_step=3, lr=1e-3, batch_size=64, buffer_size=5000,
                 sync_freq=200, beta_start=0.4, beta_frames=5000):
        self.gamma = gamma
        self.n_step = n_step
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        self.sync_freq = sync_freq

        # C51 support
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Networks
        self.net = RainbowNetwork(n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.target_net = copy.deepcopy(self.net)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # PER buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)

        # N-step buffer
        self.n_step_buffer = NStepBuffer(n_step=n_step, gamma=gamma)

        # Beta annealing for importance sampling
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

    def beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def select_action(self, state):
        """Noisy networks handle exploration — no epsilon needed!"""
        with torch.no_grad():
            q = self.net.q_values(state.unsqueeze(0))
        return q.argmax(dim=1).item()

    def store(self, transition):
        self.n_step_buffer.add(transition)
        if self.n_step_buffer.is_ready():
            n_step_transition = self.n_step_buffer.get()
            self.buffer.add(n_step_transition)

    def flush_nstep(self):
        """Flush remaining transitions in n-step buffer at episode end."""
        while len(self.n_step_buffer.buffer) > 0:
            # Build partial n-step return from what's left
            state, action, _, _, _ = self.n_step_buffer.buffer[0]
            _, _, _, next_state, done = self.n_step_buffer.buffer[-1]
            reward = 0.0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer.buffer):
                reward += (self.gamma ** i) * r
                if d:
                    next_state = self.n_step_buffer.buffer[i][-2]
                    done = True
                    break
            self.buffer.add((state, action, reward, next_state, done))
            self.n_step_buffer.buffer.popleft()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        self.frame += 1
        beta = self.beta()

        batch, indices, weights = self.buffer.sample(self.batch_size, beta)

        states = torch.stack([s for (s, a, r, s2, d) in batch])
        actions = torch.LongTensor([a for (s, a, r, s2, d) in batch])
        rewards = torch.FloatTensor([r for (s, a, r, s2, d) in batch])
        next_states = torch.stack([s2 for (s, a, r, s2, d) in batch])
        dones = torch.FloatTensor([float(d) for (s, a, r, s2, d) in batch])

        # ---- Distributional + Double DQN ----
        # Current distribution
        log_probs = self.net(states)  # (B, actions, atoms)
        log_probs_a = log_probs[range(self.batch_size), actions]  # (B, atoms)

        with torch.no_grad():
            # Double DQN: use main net to SELECT action
            next_q = self.net.q_values(next_states)
            best_actions = next_q.argmax(dim=1)

            # Use TARGET net to EVALUATE the distribution of that action
            target_log_probs = self.target_net(next_states)
            target_probs = target_log_probs.exp()
            target_probs_a = target_probs[range(self.batch_size), best_actions]  # (B, atoms)

            # Project the target distribution (Categorical projection)
            gamma_n = self.gamma ** self.n_step
            Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma_n * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Fix edge case where l == u
            l = l.clamp(0, self.n_atoms - 1)
            u = u.clamp(0, self.n_atoms - 1)

            projected = torch.zeros_like(target_probs_a)
            offset = torch.arange(self.batch_size).unsqueeze(1) * self.n_atoms

            projected.view(-1).index_add_(0, (l + offset).view(-1),
                                          (target_probs_a * (u.float() - b)).view(-1))
            projected.view(-1).index_add_(0, (u + offset).view(-1),
                                          (target_probs_a * (b - l.float())).view(-1))

        # Cross-entropy loss (KL divergence)
        loss_per_sample = -(projected * log_probs_a).sum(dim=1)
        loss = (weights * loss_per_sample).mean()

        # Update priorities in PER
        td_errors = loss_per_sample.detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Reset noise for next forward pass
        self.net.reset_noise()
        self.target_net.reset_noise()

        # Sync target network
        if self.frame % self.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return loss.item()


# ============================================================
# 6. Training Loop
# ============================================================
def test_model(agent, mode='random', n_games=100):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    wins = 0
    agent.net.eval()
    for _ in range(n_games):
        game = Gridworld(size=4, mode=mode)
        state_np = game.board.render_np().reshape(64)
        state = torch.from_numpy(state_np).float()
        for step in range(50):
            with torch.no_grad():
                q = agent.net.q_values(state.unsqueeze(0))
                action_idx = q.argmax(dim=1).item()
            game.makeMove(action_set[action_idx])
            state_np = game.board.render_np().reshape(64)
            state = torch.from_numpy(state_np).float()
            reward = game.reward()
            if reward == 10:
                wins += 1
                break
            elif reward == -10:
                break
    agent.net.train()
    return wins / n_games * 100


if __name__ == '__main__':
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    epochs = 3000
    max_moves = 50

    agent = RainbowAgent(
        n_atoms=51, v_min=-10, v_max=10,
        gamma=0.99, n_step=3, lr=5e-4,
        batch_size=64, buffer_size=5000,
        sync_freq=200, beta_start=0.4, beta_frames=epochs * 10
    )

    losses = []
    print("=" * 60)
    print("  Rainbow DQN Training on Gridworld (random mode)")
    print("  Components: Double + Dueling + PER + N-step + Noisy + C51")
    print("=" * 60)

    for epoch in range(epochs):
        game = Gridworld(size=4, mode='random')
        state_np = game.board.render_np().reshape(64)
        state = torch.from_numpy(state_np).float()
        agent.n_step_buffer.reset()

        for mov in range(max_moves):
            action_idx = agent.select_action(state)
            action = action_set[action_idx]
            game.makeMove(action)

            next_state_np = game.board.render_np().reshape(64)
            next_state = torch.from_numpy(next_state_np).float()
            reward = game.reward()
            done = (reward != -1)

            agent.store((state, action_idx, reward, next_state, done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            if done:
                break

        agent.flush_nstep()

        if (epoch + 1) % 500 == 0:
            win_rate = test_model(agent, mode='random', n_games=200)
            avg_loss = np.mean(losses[-200:]) if losses else 0
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Win Rate: {win_rate:.1f}%")

    # Final evaluation
    final_win_rate = test_model(agent, mode='random', n_games=500)
    print("\n" + "=" * 60)
    print(f"  Final Win Rate (500 games, random mode): {final_win_rate:.1f}%")
    print("=" * 60)
