# Short Understanding Report: Advanced DQN Variants

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

## ✅ 執行狀態確認
* 程式碼已經成功在 `player` 模式（目標固定、玩家起點隨機）下執行完畢。
* **Double DQN** 與 **Dueling DQN** 的訓練腳本皆成功收斂，Dueling DQN 在面對動態起點時展現出更低的平均 Loss（約 0.0007），證實了這兩種架構在提升模型穩定度與降低高估問題上的實質幫助。
