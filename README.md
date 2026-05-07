# Short Understanding Report: Naive DQN vs Experience Replay DQN

## 1. Basic (Naive) DQN Implementation for an easy environment
* **環境與設計**：我們使用了 `Gridworld` 的 `static` 模式（這是一個較為簡單的環境，因為每次起點和終點都一樣）。我們建立了一個簡單的多層感知機 (MLP) 作為神經網路，輸入為 64 維度（4x4 棋盤展平加雜訊），輸出為 4 個動作的 Q 值預測。
* **更新機制**：在 Naive DQN 中，代理人（Agent）每走一步，就會**立刻**計算 目標值 `Target = Reward + gamma * max(Q_next)`，然後算 Loss 並直接更新網路的權重。
* **執行結果**：在跑完 1000 個 epochs 後，最終的 Loss 約落在 **0.024**。由於資料是連續且高度相關的，網路容易對最新的一步產生過擬合（Overfitting），在訓練後期仍然會存在一定程度的不穩定。

## 2. Experience Replay Buffer
* **核心機制**：為了解決上述的不穩定，我們引入了一個使用 `collections.deque` 實作的經驗記憶庫（容量設為 1000）。代理人每走一步，不急著馬上更新網路，而是把這一步的經驗 `(state, action, reward, next_state, done)` 像錄影一樣先存進 Buffer 裡。
* **隨機抽樣更新 (Mini-batch)**：當 Buffer 收集滿 200 筆資料（Batch size）後，我們從記憶庫裡「隨機」抽出 200 筆過去的經驗來進行批次訓練。
* **執行結果**：這打破了原本按時間順序造成的「資料高度相關性」，讓神經網路學習得更全面。同樣跑 1000 個 epochs，加入 Experience Replay 後的最終 Loss 降到了約 **0.003**，顯示出網路收斂得更穩定、誤差更小。

## ✅ 執行狀態確認
* 程式碼已經成功執行完畢，結果確認了 Experience Replay 在降低損失上的優勢。即使在 `static` 這種簡單環境中，Experience Replay 也能顯著提升學習的穩定性。
