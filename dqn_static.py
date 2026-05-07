import numpy as np
import torch
from Gridworld import Gridworld
import random

def train_naive_dqn_with_replay(mode='static', epochs=1000, use_replay=True):
    # Network dimensions
    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    # Build the Neural Network
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 1.0

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }

    losses = []
    
    # Experience Replay settings
    from collections import deque
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50

    print(f"Starting training on Gridworld (mode={mode}) with Experience Replay: {use_replay}")
    
    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0
        
        while(status == 1):
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            
            if not use_replay:
                # Basic / Naive DQN update
                with torch.no_grad():
                    newQ = model(state2)
                maxQ = torch.max(newQ)
                if reward == -1:
                    Y = reward + (gamma * maxQ)
                else:
                    Y = reward
                Y = torch.Tensor([Y]).detach()
                X = qval.squeeze()[action_]
                loss = loss_fn(X, Y)
                
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                state1 = state2
                
                if reward != -1 or mov > max_moves:
                    status = 0
            else:
                # DQN with Experience Replay Buffer
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
                        Q2 = model(state2_batch)
                    
                    Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = loss_fn(X, Y.detach())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    
                if reward != -1 or mov > max_moves:
                    status = 0
                    
        if epsilon > 0.1:
            epsilon -= (1 / epochs)
            
    print(f"Training Complete! Final Loss: {losses[-1] if len(losses) > 0 else 'N/A'}")
    return model

if __name__ == '__main__':
    # Run Basic (Naive) DQN on Static mode
    print("--- Running Basic DQN on Static Mode ---")
    model_naive = train_naive_dqn_with_replay(mode='static', epochs=1000, use_replay=False)
    
    # Run DQN with Experience Replay on Static mode
    print("\n--- Running DQN with Experience Replay on Static Mode ---")
    model_replay = train_naive_dqn_with_replay(mode='static', epochs=1000, use_replay=True)
