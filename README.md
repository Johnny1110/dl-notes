# DL Notes

<br>

---

<br>

## Before You Start

run `pip install -r requirements.txt` to install all the dependencies.

<br>
<br>

## Road Map


* **Phase 0**: Python + 數學基礎
    
* **Phase 1**: 神經網路基礎（手刻理解原理）
    
* **Phase 2**: Deep Learning 核心架構（CNN, RNN）
    
* **Phase 3**: Transformer 深入理解
    
* **Phase 4**: LLM 原理與訓練
    
* **Phase 5**: LLM 應用工程（RAG, Fine-tuning, 部署）
    
* **Phase 6**: 專題 — 自己訓練並部署一個完整的 LLM 應用


<br>

---

<br>

## Phase 0：Python 與數學基礎

### 目標

你不需要成為數學家，但需要能「讀懂公式並理解它在程式裡對應什麼操作」。

### Python

重點放在：

- NumPy 的矩陣操作（這是後面所有框架的底層邏輯）
- Matplotlib 基本繪圖（用來視覺化訓練過程）
- Jupyter Notebook 的使用習慣（ML 領域的標準實驗環境）

<br>

### 數學：線性代數

| 主題 | 為什麼需要 |
|------|-----------|
| 向量與矩陣運算（加法、乘法、轉置） | 神經網路的每一層本質上就是矩陣乘法 |
| 點積（Dot Product） | Attention 機制的核心就是向量的點積 |
| 矩陣的形狀（Shape）與廣播（Broadcasting） | 你會花大量時間 debug tensor shape 不匹配的問題 |
| 特徵值與特徵向量（概念即可） | 理解 PCA、理解為什麼某些矩陣操作有效 |

<br>

[線性代數 - 學習入口](phase_0/linear_algebra)

<br>

### 數學：微積分

| 主題 | 為什麼需要 |
|------|-----------|
| 導數 / 偏微分 | 梯度下降的基礎，告訴模型「往哪個方向調整」 |
| 鏈式法則（Chain Rule） | 反向傳播（Backpropagation）的數學基礎 |
| 梯度（Gradient） | 多維空間中的導數，控制模型參數更新的方向和幅度 |

<br>

[微積分 - 學習入口](phase_0/calculus)

<br>

### 數學：機率與統計

| 主題 | 為什麼需要 |
|------|-----------|
| 機率分佈（常態分佈、均勻分佈） | 權重初始化、資料分佈的理解 |
| 條件機率、貝氏定理 | 理解語言模型的生成邏輯：P(下一個字 \| 前面的字) |
| Softmax 函數 | 把任意數值轉成機率分佈，分類問題和 Attention 都會用到 |

<br>

### 推薦資源

- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)（視覺化直覺，強烈推薦）
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [Khan Academy — 機率與統計](https://www.khanacademy.org/math/statistics-probability)（按需查閱）

### 驗收標準

- 能用 NumPy 手動實作矩陣乘法、轉置
- 能解釋什麼是梯度，以及它跟「學習」的關係
- 能手算一個簡單函數的偏微分並用鏈式法則推導
- 能解釋 Softmax 函數做了什麼事、為什麼輸出總和為 1

<br>

---

<br>

## Phase 1：神經網路基礎

### 目標

理解神經網路是怎麼「學習」的。這個階段的核心是 **Backpropagation**——搞懂這個，後面所有的東西都是它的延伸。

<br>

### 核心概念

**神經元與前向傳播（Forward Pass）**
- 一個神經元做的事：`output = activation(weights · inputs + bias)`
- 多層串接：每一層的輸出是下一層的輸入
- 這本質上就是一連串矩陣乘法 + 非線性函數

**損失函數（Loss Function）**
- 衡量「模型的預測」和「正確答案」差多遠
- MSE（均方誤差）：回歸問題
- Cross-Entropy：分類問題（後面語言模型也用這個）

**反向傳播（Backpropagation）**
- 用鏈式法則計算每個參數對 loss 的影響程度（梯度）
- 這就是為什麼 Phase 0 要學鏈式法則

**梯度下降（Gradient Descent）與優化器**
- SGD：最基本的，每次用一小批資料更新參數
- Adam：目前最常用，自動調整學習率
- 學習率（Learning Rate）：太大會震盪，太小會卡住

**過擬合（Overfitting）與正則化**
- 訓練集表現很好，測試集很差 = 過擬合
- Dropout：訓練時隨機關掉一些神經元，迫使網路學到更泛化的特徵
- Weight Decay：限制參數不要太大

**激活函數（Activation Functions）**
- ReLU：目前最常用，解決了梯度消失問題
- Sigmoid / Tanh：歷史原因需要知道，某些場景還會用

### 實作任務

1. **手刻神經網路（不用任何框架）**
   - 只用 NumPy 實作一個 2 層神經網路
   - 用它做 MNIST 手寫數字辨識
   - 自己寫 forward pass、loss 計算、backward pass、參數更新
   - 這個練習的價值極高，做完之後你對神經網路就不再是黑盒了

2. **用 PyTorch 重寫一次**
   - 學會 `torch.Tensor`、`autograd`、`nn.Module`
   - 理解框架幫你自動做了哪些事（主要是自動微分）
   - 學會訓練迴圈的標準寫法：`forward → loss → backward → step`

### 推薦資源

- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)（從零手刻）
- [PyTorch 官方教程](https://pytorch.org/tutorials/)

### 驗收標準

- 能不看任何資料，用 NumPy 手刻一個可以訓練的神經網路
- 能清楚解釋 Backpropagation 在做什麼（不只是概念，是數學流程）
- 能用 PyTorch 訓練一個 MNIST 分類器並達到 >95% 準確率
- 能解釋 overfitting 是什麼、怎麼發現、怎麼處理

<br>

[Neural Networks: Zero to Hero - 學習入口](phase_1/neural_networks_zero_2_here)

<br>

---

<br>

## Phase 2：Deep Learning 核心架構

### 目標

理解 Transformer 之前的兩大架構，知道它們各自解決了什麼問題、有什麼限制。這會讓你理解 **Transformer 為什麼被發明出來**。

<br>

### CNN（卷積神經網路）

**解決的問題**：圖片辨識。全連接神經網路處理圖片時參數量爆炸，CNN 利用「局部特徵」的概念大幅減少參數。

| 概念 | 說明 |
|------|------|
| 卷積核（Kernel/Filter） | 一個小矩陣在圖片上滑動，提取局部特徵（邊緣、紋理） |
| Pooling | 降維，保留重要特徵同時減少計算量 |
| Feature Map | 卷積後的輸出，代表「模型看到了什麼」 |
| 經典架構 | LeNet → AlexNet → VGG → ResNet（理解演進邏輯） |

**你需要知道但不用精通**：CNN 不是你的目標方向，但 Transformer 後來在視覺領域（ViT）取代了 CNN，理解 CNN 的限制能幫你理解為什麼。

### RNN / LSTM（循環神經網路）

**解決的問題**：序列資料（文字、時間序列）。前面的輸入需要影響後面的輸出。

| 概念 | 說明 |
|------|------|
| 隱藏狀態（Hidden State） | RNN 的「記憶」，把前面的資訊壓縮成一個向量往後傳 |
| 梯度消失 / 梯度爆炸 | RNN 的致命問題：序列一長，梯度要麼趨近於零要麼爆炸 |
| LSTM | 用 Gate 機制（遺忘門、輸入門、輸出門）來控制記憶的保留和丟棄 |
| Seq2Seq | 編碼器-解碼器架構，機器翻譯的經典模型 |

**為什麼重要**：RNN/LSTM 是 Transformer 的「前任」。Transformer 的核心創新——Self-Attention——就是為了解決 RNN 的兩個致命問題：
1. 無法平行計算（必須按順序處理序列）
2. 長距離依賴問題（早期的資訊會被遺忘）

### 實作任務

1. **CNN**：用 PyTorch 搭一個 CNN 做 CIFAR-10 圖片分類
2. **RNN/LSTM**：用 LSTM 做一個簡單的文字生成（給定前幾個字，預測下一個字）
3. **Seq2Seq**：實作一個簡單的序列到序列模型（例如日期格式轉換）

### 驗收標準

- 能解釋 CNN 的卷積操作在數學上做了什麼
- 能解釋 RNN 的梯度消失問題，以及 LSTM 如何緩解它
- 能清楚說出「為什麼 RNN 不能平行計算但 Transformer 可以」



<br>

---

<br>

## Phase 3：Transformer 深入理解

### 目標

這是整個路線圖的**核心中的核心**。你需要理解 Transformer 到能自己從零實作的程度。

<br>

### 核心概念

**Attention 機制的直覺**
- 核心問題：當模型在處理一個 token 時，它應該「注意」序列中的哪些其他 token？
- Self-Attention 讓每個 token 自己決定要關注誰

**Q、K、V（Query, Key, Value）**
- 用你的後端經驗來理解：這就像一個 key-value store 的查詢
- Query = 你想查什麼
- Key = 索引
- Value = 實際的內容
- 相似度 = Query 和 Key 的點積

**Scaled Dot-Product Attention**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- 為什麼要除以 $\sqrt{d_k}$？防止點積值太大導致 softmax 輸出趨近 one-hot
- 這個公式你需要能閉眼寫出來

**Multi-Head Attention**
- 不只做一次 attention，同時做多次（多個 head），每個 head 關注不同的特徵
- 最後把所有 head 的輸出拼接起來

**位置編碼（Positional Encoding）**
- Transformer 沒有像 RNN 那樣的順序概念，所以需要額外注入位置資訊
- 原始論文用 sin/cos，現代模型用 RoPE（Rotary Position Embedding）

**完整的 Transformer Block**

```
Input
  → Multi-Head Self-Attention
  → Add & LayerNorm（殘差連接 + 層正規化）
  → Feed-Forward Network（兩層全連接）
  → Add & LayerNorm
Output
```

**Encoder vs Decoder**
- Encoder：雙向注意力，能看到整個序列（BERT 用這個）
- Decoder：帶 Mask 的單向注意力，只能看到前面的 token（GPT 用這個）
- 原始 Transformer 是 Encoder-Decoder 架構（翻譯任務用）

### 延伸主題

| 主題 | 說明 |
|------|------|
| RoPE | 旋轉位置編碼，目前主流 LLM 的標準配備 |
| FlashAttention | 不改變數學結果，透過 IO 優化大幅加速 Attention 計算 |
| KV Cache | 推論時避免重複計算歷史 token 的 Key 和 Value |
| GQA（Grouped-Query Attention） | 多個 Query Head 共享 Key/Value，減少記憶體佔用 |

### 必讀論文

1. [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — Transformer 的原始論文，必讀
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 視覺化解說，幫助建立直覺

### 實作任務

1. **從零實作 Transformer**：只用 PyTorch 的基本 tensor 操作，不用 `nn.Transformer`
   - 實作 Scaled Dot-Product Attention
   - 實作 Multi-Head Attention
   - 實作完整的 Transformer Block
   - 實作 Positional Encoding
2. **用你的 Transformer 做一個簡單任務**（例如小型翻譯或文字生成）

### 推薦資源

- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（從零手刻 GPT，2 小時）

### 驗收標準

- 能在白板上畫出完整的 Transformer 架構圖
- 能不看資料寫出 Self-Attention 的公式並解釋每一項
- 能從零實作一個可以訓練的 Transformer
- 能解釋 Encoder 和 Decoder 的差異，以及為什麼 GPT 只用 Decoder


<br>

---

<br>

## Phase 4：LLM 原理與訓練

### 目標

理解大型語言模型是怎麼從 Transformer 架構發展出來的，以及訓練一個 LLM 需要什麼。

### 預訓練（Pre-training）

**GPT 系列（Decoder-only, Autoregressive）**
- 訓練目標：給定前面的 token，預測下一個 token
- 這就是一個超大規模的「完形填空」，但只填後面
- 損失函數：Cross-Entropy Loss

**BERT（Encoder-only, Masked Language Model）**
- 訓練目標：隨機遮住一些 token，讓模型預測被遮住的是什麼
- 雙向注意力，適合理解型任務（分類、問答）
- 了解即可，目前 LLM 主流是 GPT 風格的 Decoder-only

### Tokenization

| 方法 | 說明 |
|------|------|
| BPE（Byte-Pair Encoding） | GPT 系列使用，把常見的字元組合合併成一個 token |
| WordPiece | BERT 使用 |
| SentencePiece | 支援多語言，不依賴空格分詞 |

為什麼重要：Tokenization 決定了模型「看到」什麼，也影響上下文長度的利用效率。

### Scaling Laws

- 模型越大、資料越多、算力越強 → 效果越好（在一定範圍內）
- Chinchilla 論文指出：大部分模型都 undertrained（資料量不夠）
- 理解 Scaling Laws 能幫你判斷「該加大模型還是加多資料」

### RLHF / Alignment

- Pre-training 之後的模型很強但不受控（會輸出有害內容）
- SFT（Supervised Fine-Tuning）：用人類標註的對話資料做微調
- RLHF（Reinforcement Learning from Human Feedback）：訓練一個獎勵模型，再用強化學習調整 LLM
- DPO（Direct Preference Optimization）：RLHF 的簡化版，不需要獎勵模型
- 這是 ChatGPT 能「聽話」的關鍵技術

### 核心工程主題

| 主題 | 說明 |
|------|------|
| 分散式訓練 | 一張 GPU 裝不下的模型怎麼訓練？Data Parallelism、Model Parallelism、Pipeline Parallelism |
| Mixed Precision Training | 用 FP16/BF16 加速訓練同時減少記憶體 |
| Gradient Checkpointing | 用計算換記憶體，降低訓練時的 VRAM 需求 |
| DeepSpeed / FSDP | 分散式訓練框架 |

### 推薦閱讀

- [GPT-2 論文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [LLaMA 論文](https://arxiv.org/abs/2302.13971) — Meta 的開源 LLM，架構講解清晰
- [InstructGPT 論文](https://arxiv.org/abs/2203.02155) — RLHF 的實作細節

### 驗收標準

- 能解釋 GPT 和 BERT 的訓練目標有什麼不同
- 能解釋 BPE tokenization 的演算法步驟
- 能解釋 RLHF 的完整流程（SFT → 獎勵模型 → PPO）
- 能解釋為什麼訓練 LLM 需要分散式系統

<br>

---

<br>

## Phase 5：LLM 應用工程

### 目標

把前面學的理論應用到實際工程中。這個階段你的後端經驗會成為巨大優勢。

### Prompt Engineering

- System Prompt / User Prompt / Assistant Prompt 的設計
- Few-shot Learning：在 prompt 裡給範例
- Chain of Thought：引導模型逐步推理
- Output Formatting：控制輸出格式（JSON Schema）

### RAG（Retrieval-Augmented Generation）

**完整的 RAG Pipeline：**

```
文件 → 分塊（Chunking）→ Embedding → 存入向量資料庫
                                            ↓
使用者提問 → Embedding → 向量搜尋 → 取出相關片段 → 組合成 Prompt → LLM 生成回答
```

| 元件 | 技術選型 |
|------|---------|
| Embedding 模型 | OpenAI text-embedding, BGE, E5 |
| 向量資料庫 | pgvector（你會的 PostgreSQL 加個擴充）、Pinecone、Milvus |
| 分塊策略 | Fixed-size, Recursive, Semantic Chunking |
| Reranking | 用 Cross-Encoder 對搜尋結果重新排序 |

### Fine-tuning

| 方法 | 說明 |
|------|------|
| Full Fine-tuning | 更新所有參數，需要大量 GPU，通常不實際 |
| LoRA | 只訓練低秩矩陣（佔原參數的 <1%），效果接近 full fine-tuning |
| QLoRA | LoRA + 4-bit 量化，單張消費級 GPU 就能跑 |

### 模型部署與推論優化

| 技術 | 說明 |
|------|------|
| 量化（Quantization） | 把 FP16 壓成 INT8/INT4，模型變小、推論變快 |
| vLLM | 高效推論框架，支援 Paged Attention、Continuous Batching |
| TGI（Text Generation Inference） | HuggingFace 的推論伺服器 |
| Streaming | Token-by-token 輸出，減少首字延遲（TTFT） |
| Batching 策略 | 怎麼最大化 GPU 利用率 |

### 實作任務

1. **搭建一個 RAG 系統**：用 pgvector + LangChain/LlamaIndex，對一組技術文件做問答
2. **用 QLoRA 微調一個模型**：例如讓 Llama 用特定風格或格式回答問題
3. **部署一個推論服務**：用 vLLM 部署開源模型，掛上 API Gateway，做好監控

### 驗收標準

- 能設計並實作一個完整的 RAG pipeline
- 能用 QLoRA 微調一個 7B 參數的模型
- 能用 vLLM 部署模型並處理並發請求
- 能解釋量化對模型品質的影響

<br>

---

<br>

## Phase 6：畢業專題

### 目標

整合所有學到的知識，做一個**端到端的完整專案**，從資料處理到模型訓練到部署上線。

### 專案建議：「針對特定領域的 LLM 問答系統」

> 例如：一個針對某個開源專案（Kubernetes / Spring Boot / Go 標準庫）文件的智慧問答系統。

**你需要做的事：**

```
1. 資料收集與處理
   - 爬取目標領域的文件
   - 清洗、分塊、建立 Embedding 索引

2. 模型層
   - 選擇基礎模型（Llama / Mistral）
   - 用 QLoRA 針對該領域的 QA 資料做 Fine-tuning
   - 設計 evaluation pipeline 衡量模型品質

3. 檢索層（RAG）
   - 向量資料庫的選型與建置
   - Chunking 策略的實驗與比較
   - Reranking 機制

4. 服務層（你的主場）
   - 用 vLLM 部署推論服務
   - 設計 API（REST / gRPC）
   - 實作 streaming response
   - 監控：延遲、吞吐量、GPU 利用率
   - 如果想挑戰：做 auto-scaling

5. 前端（選做）
   - 簡單的 chat UI
   - 顯示引用來源
```

**為什麼這個專題好：**
- 涵蓋了 Phase 1-5 的所有知識
- RAG + Fine-tuning + 部署的完整組合
- 後端服務設計是你的強項，可以展示差異化能力
- 有實際使用價值，可以放在履歷和 GitHub 上

### 驗收標準

- 系統能穩定運行並回答領域內問題
- 回答品質明顯優於直接用基礎模型（可量化比較）
- 完整的 API 文件
- 有監控與日誌
- README 清楚說明架構設計與技術選型的理由

<br>

---

<br>

## 附錄：推薦學習資源總整理

### 影片課程

| 資源 | 適用階段 | 說明 |
|------|---------|------|
| [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Phase 0-1 | 視覺化直覺，極佳入門 |
| [Andrej Karpathy — Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Phase 1-3 | 從零手刻，最適合想理解原理的工程師 |
| [Stanford CS231n](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Agt3css-6) | Phase 2 | CNN 的經典課程 |
| [Stanford CS224n](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4) | Phase 2-3 | NLP 與 Transformer |


### 工具

| 工具 | 用途 |
|------|------|
| PyTorch | 主力框架，必學 |
| Hugging Face Transformers | 預訓練模型的標準庫 |
| Weights & Biases (wandb) | 實驗追蹤與視覺化 |
| LangChain / LlamaIndex | RAG 框架 |
| vLLM | 推論部署 |

---

> 💡 **給自己的提醒**：不要為了趕進度而跳過「手刻實作」。理解原理比跑得快重要。遇到卡關的時候，回頭重看前一個階段的內容，通常是基礎不夠扎實。

<br>
<br>

---

<br>
<br>

## 隨記

* [From attention to agent Era](gossip/attention2agent)
