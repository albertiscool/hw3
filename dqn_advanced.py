import numpy as np
import torch
import copy
from collections import deque
import random
from Gridworld import Gridworld

# Standard Sequential Model (for Double DQN)
def create_standard_model():
    return torch.nn.Sequential(
        torch.nn.Linear(64, 150),
        torch.nn.ReLU(),
        torch.nn.Linear(150, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 4)
    )

# Dueling DQN Model
class DuelingDQN(torch.nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(64, 150)
        self.relu1 = torch.nn.ReLU()
        
        self.val_fc = torch.nn.Linear(150, 100)
        self.val_relu = torch.nn.ReLU()
        self.val_out = torch.nn.Linear(100, 1)
        
        self.adv_fc = torch.nn.Linear(150, 100)
        self.adv_relu = torch.nn.ReLU()
        self.adv_out = torch.nn.Linear(100, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        
        val = self.val_relu(self.val_fc(x))
        val = self.val_out(val)
        
        adv = self.adv_relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

def train_agent(mode='player', epochs=2000, use_double=False, use_dueling=False):
    print(f"--- Training Gridworld ({mode} mode) ---")
    print(f"Double DQN: {use_double} | Dueling DQN: {use_dueling}")
    
    if use_dueling:
        model = DuelingDQN()
    else:
        model = create_standard_model()
        
    target_model = copy.deepcopy(model)
    target_model.load_state_dict(model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 1.0
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses = []
    
    mem_size = 2000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    sync_freq = 500
    step_count = 0

    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while(status == 1):
            step_count += 1
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            
            done = True if reward > 0 else False
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                Q1 = model(state1_batch)
                
                with torch.no_grad():
                    if use_double:
                        # Double DQN: Main model selects action, Target model evaluates
                        Q2_main = model(state2_batch)
                        best_actions = torch.argmax(Q2_main, dim=1)
                        Q2_target = target_model(state2_batch)
                        Q2_max = Q2_target.gather(1, best_actions.unsqueeze(1)).squeeze()
                    else:
                        # Standard Target DQN
                        Q2_target = target_model(state2_batch)
                        Q2_max = torch.max(Q2_target, dim=1)[0]
                
                Y = reward_batch + gamma * ((1 - done_batch) * Q2_max)
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(1)).squeeze()
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if step_count % sync_freq == 0:
                    target_model.load_state_dict(model.state_dict())
                    
            if reward != -1 or mov > max_moves:
                status = 0
                
        if epsilon > 0.1:
            epsilon -= (1 / epochs)
            
    final_loss = sum(losses[-100:])/100 if len(losses) >= 100 else 0
    print(f"Training Complete! Average Loss over last 100 steps: {final_loss:.4f}\n")
    return model

if __name__ == '__main__':
    # 1. Standard target network (Baseline)
    train_agent(mode='player', epochs=1000, use_double=False, use_dueling=False)
    
    # 2. Double DQN
    train_agent(mode='player', epochs=1000, use_double=True, use_dueling=False)
    
    # 3. Dueling DQN
    train_agent(mode='player', epochs=1000, use_double=False, use_dueling=True)
