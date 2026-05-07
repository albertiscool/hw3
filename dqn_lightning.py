import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import random
from collections import deque
from Gridworld import Gridworld

class RLDataset(IterableDataset):
    def __init__(self, buffer, epoch_size=1000):
        self.buffer = buffer
        self.epoch_size = epoch_size

    def __iter__(self):
        # Provide a stream of random samples from the buffer
        for _ in range(self.epoch_size):
            yield random.choice(self.buffer)

class LitDQN(pl.LightningModule):
    def __init__(self, mode='player', batch_size=200, lr=1e-3, gamma=0.9, sync_rate=500):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Core Networks
        self.net = torch.nn.Sequential(
            torch.nn.Linear(64, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 4)
        )
        self.target_net = torch.nn.Sequential(
            torch.nn.Linear(64, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 4)
        )
        self.target_net.load_state_dict(self.net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        # 2. Environment & Buffer
        self.buffer = deque(maxlen=2000)
        self.env = Gridworld(size=4, mode=mode)
        self.epsilon = 1.0
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.reset_env()
        
        self.loss_fn = torch.nn.MSELoss()
        
    def reset_env(self):
        self.env = Gridworld(size=4, mode=self.hparams.mode)
        state_np = self.env.board.render_np().reshape(64) + np.random.rand(64) / 100.0
        self.state = torch.from_numpy(state_np).float()

    def populate_buffer(self, steps=1000):
        print("Populating initial buffer...")
        for _ in range(steps):
            self.play_step(force_random=True)

    def forward(self, x):
        return self.net(x)

    def play_step(self, force_random=False):
        if force_random or random.random() < self.epsilon:
            action_idx = np.random.randint(0, 4)
        else:
            q_vals = self.net(self.state.unsqueeze(0))
            action_idx = torch.argmax(q_vals).item()

        action = self.action_set[action_idx]
        self.env.makeMove(action)
        
        next_state_np = self.env.board.render_np().reshape(64) + np.random.rand(64) / 100.0
        next_state = torch.from_numpy(next_state_np).float()
        reward = self.env.reward()
        done = True if reward > 0 or reward == -10 else False
        
        exp = (self.state, action_idx, reward, next_state, done)
        self.buffer.append(exp)
        
        self.state = next_state
        if done:
            self.reset_env()
            
    def training_step(self, batch, batch_idx):
        # Decrease epsilon over time
        self.epsilon = max(0.1, self.epsilon - 1e-4)
        
        # Agent plays a step in the environment to collect data
        self.play_step()
        
        states, actions, rewards, next_states, dones = batch
        
        q_vals = self.net(states)
        q_val = q_vals.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            target_q_vals = self.target_net(next_states)
            max_target_q_val = target_q_vals.max(1)[0]
            target = rewards + self.hparams.gamma * (1 - dones.float()) * max_target_q_val
            
        loss = self.loss_fn(q_val, target)
        
        # Target Network Synchronization
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        # Learning Rate Scheduler (Bonus feature to stabilize learning)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, epoch_size=1000)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

if __name__ == '__main__':
    model = LitDQN(mode='random')
    model.populate_buffer(1000)
    
    # PyTorch Lightning Trainer with Gradient Clipping (Bonus feature)
    trainer = pl.Trainer(
        max_epochs=5,
        gradient_clip_val=1.0, # Gradient clipping to prevent exploding gradients
        enable_checkpointing=False,
        logger=False
    )
    
    print("Starting PyTorch Lightning Training with Gradient Clipping & LR Scheduling...")
    trainer.fit(model)
    print("Lightning Training Complete!")
