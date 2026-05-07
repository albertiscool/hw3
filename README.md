# Short Understanding Report: Advanced DQN Variants & Framework Migration

## 1. Basic (Naive) DQN Implementation for an easy environment
* **環境與設計**：我們使用了 `Gridworld` 的 `static` 模式（這是一個較為簡單的環境，因為每次起點和終點都一樣）。我們建立了一個簡單的多層感知機 (MLP) 作為神經網路，輸入為 64 維度（4x4 棋盤展平加雜訊），輸出為 4 個動作的 Q 值預測。
* **更新機制**：在 Naive DQN 中，代理人（Agent）每走一步，就會**立刻**計算 目標值 `Target = Reward + gamma * max(Q_next)`，然後算 Loss 並直接更新網路的權重。
* **執行結果**：在跑完 1000 個 epochs 後，最終的 Loss 約落在 **0.024**。由於資料是連續且高度相關的，網路容易對最新的一步產生過擬合（Overfitting），在訓練後期仍然會存在一定程度的不穩定。

## 2. Experience Replay Buffer
* **核心機制**：為了解決上述的不穩定，我們引入了一個使用 `collections.deque` 實作的經驗記憶庫（容量設為 1000）。代理人每走一步，不急著馬上更新網路，而是把這一步的經驗 `(state, action, reward, next_state, done)` 像錄影一樣先存進 Buffer 裡。
* **隨機抽樣更新 (Mini-batch)**：當 Buffer 收集滿 200 筆資料（Batch size）後，我們從記憶庫裡「隨機」抽出 200 筆過去的經驗來進行批次訓練。
* **執行結果**：這打破了原本按時間順序造成的「資料高度相關性」，讓神經網路學習得更全面。同樣跑 1000 個 epochs，加入 Experience Replay 後的最終 Loss 降到了約 **0.003**，顯示出網路收斂得更穩定、誤差更小。

---

## 3. Double DQN
* **解決的問題（Overestimation）**：在 Basic DQN 中，計算 Target 時會直接使用 `max(Q_next)`。這意味著我們用**同一個網路**來「選擇動作」和「評估動作價值」。當網路預測有誤差時，取 Max 會讓這個誤差被**過度高估 (Overestimation)**。
* **如何改進**：Double DQN 將「選擇動作」和「評估價值」拆開：
  1. 使用**主網路 (Main Network)** 來決定下一個狀態中最好的動作。
  2. 使用**目標網路 (Target Network)** 來計算這個動作的 Q 值。
  這樣可以大幅減少高估誤差，讓模型對真正的好壞判斷更加精準。

## 4. Dueling DQN
* **解決的問題（不必要的動作評估）**：在很多遊戲狀態下，無論你做什麼動作，結果都差不多（例如：前方有一大段空路，走哪都一樣）。標準的 DQN 會去學習每一個 $(State, Action)$ 的絕對價值。
* **如何改進**：Dueling DQN 在神經網路的末端**分流為兩條支線 (Streams)**：
  1. **Value Stream $V(s)$**：評估「現在這個狀態本身有多好」。
  2. **Advantage Stream $A(s, a)$**：評估「在這個狀態下，採取某個特定動作比平均狀況好多少」。
  最後將這兩者結合得到 $Q(s, a) = V(s) + (A(s, a) - \text{mean}(A(s, a)))$。
  這讓模型在面臨許多無關緊要的選擇時，只需單純學習一次 $V(s)$ 即可，不需重複學習每個動作的 Q 值，學習效率顯著提高。

---

## 5. Migration to PyTorch Lightning & Advanced Training Techniques (Bonus)
為了讓整個強化學習訓練的程式碼更簡潔、模組化，並且更容易擴充，我們將原本純 PyTorch 的迴圈改寫並遷移到了 **PyTorch Lightning** 框架中。

* **LightningModule 整合**：我們設計了 `LitDQN`，將網路定義、資料採集 (Environment Steps)、優化器與 Target Network 的同步，全部封裝在一個類別中。並且透過實作 `IterableDataset` 讓 Lightning 的 `Trainer` 能夠優雅地接管 Experience Replay Buffer。
* **Bonus - Gradient Clipping (梯度截斷)**：在 RL 中（特別是遇到巨大懲罰或獎勵時），Loss 的突增可能會導致神經網路權重崩潰。在 PyTorch Lightning 中，我們只要在 `Trainer` 加入一行 `gradient_clip_val=1.0`，就能輕鬆防止 Exploding Gradients 問題。
* **Bonus - Learning Rate Scheduling (學習率排程)**：初期我們希望模型大步探索，後期則需要細微的調整。透過 Lightning 的 `configure_optimizers`，我們整合了 `StepLR`，讓學習率在訓練中後期自動衰減 (Gamma=0.9)，使訓練收斂更為穩定。

## ✅ 執行狀態確認
* **Basic Naive DQN** (`static` 模式)：1000 epochs，最終 Loss ≈ 0.024。
* **DQN + Experience Replay** (`static` 模式)：1000 epochs，最終 Loss ≈ 0.003，收斂更穩定。
* **Double DQN** 與 **Dueling DQN** (`player` 模式，玩家起點隨機)：Dueling DQN 平均 Loss 最低（約 0.0007），有效改善 Q 值高估問題。
* **PyTorch Lightning** (`random` 模式，**最難等級**，所有物件全部隨機擺放)：透過 `dqn_lightning.py` 成功執行，Gradient Clipping (`gradient_clip_val=1.0`) 與 Learning Rate Scheduling (`StepLR`) 皆正常運作，在最複雜的環境中依然保持穩定學習。
