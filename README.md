# Stable Diffusion Web UI

一個基於 Streamlit 的 Stable Diffusion Web UI 介面，支援多種模型載入、LoRA 模型應用、圖片生成和後期處理功能。

## 功能特點

- 支援多種 Stable Diffusion 模型載入
- 支援 LoRA 模型的載入和權重調整
- 圖片生成參數完整控制
- 提供多種採樣方法
- 內建圖片後期處理功能
- 自動保存生成記錄
- 支援圖片縮放和預覽
- 支援浮水印添加
- 提供多種提示詞模板

## 系統需求

- Python 3.8 或更高版本
- CUDA 相容的 GPU（推薦 8GB 以上顯存）
- 至少 16GB 系統記憶體

## 安裝指南

1. 克隆專案倉庫：
```bash
git clone [repository-url]
cd stable-diffusion-webui
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 準備模型：
   - 在專案根目錄創建 `model` 目錄
   - 將 Stable Diffusion 模型放入 `model` 目錄
   - 若要使用 LoRA，創建 `lora` 目錄並放入 LoRA 模型

## 目錄結構

```
stable-diffusion-webui/
├── app.py                 # 主程式
├── utils/
│   ├── model_utils.py     # 模型相關工具
│   ├── image_utils.py     # 圖片處理工具
│   └── ui_utils.py        # UI 介面工具
├── model/                 # 基礎模型目錄
├── lora/                  # LoRA 模型目錄
└── outputs/              # 生成圖片輸出目錄
```

## 使用方法

1. 啟動應用程式：
```bash
streamlit run app.py
```

2. 基本操作流程：
   - 選擇基礎模型
   - 選擇並設定 LoRA 模型（可選）
   - 調整生成參數
   - 輸入提示詞
   - 點擊生成按鈕

### 模型管理

- 支援的模型格式：
  - 目錄型模型（包含 model_index.json 或 config.json）
  - 單檔案模型（.ckpt, .safetensors, .bin, .pth）
- LoRA 模型支援格式：.safetensors, .pt, .bin

### 參數說明

- **寬度/高度**：生成圖片的尺寸（256-1024）
- **CFG Scale**：提示詞引導強度（1.0-20.0）
- **Sampling Steps**：生成步數（1-150）
- **Seed**：隨機種子（-1 為隨機）
- **Sampling Method**：多種採樣方法可選

### LoRA 使用說明

- 支援同時使用多個 LoRA 模型
- 每個 LoRA 可獨立調整權重（0.0-2.0）
  - 1.0：正常強度
  - <1.0：降低效果
  - >1.0：增強效果

### 圖片後期處理

支援以下調整：
- 對比度
- 色溫
- 銳利度
- RGB 通道
- 飽和度
- 亮部
- 暗部

## 注意事項

1. 顯存使用：
   - 建議使用 8GB 以上顯存的 GPU
   - 使用多個 LoRA 會增加顯存使用量
   - 較大的圖片尺寸需要更多顯存

2. 效能優化：
   - 啟用了記憶體優化功能
   - 支援半精度（FP16）運算
   - 使用 attention slicing 和 VAE tiling

3. 檔案管理：
   - 自動保存功能預設開啟
   - 生成的圖片會保存在 outputs 目錄
   - 支援下載原始和調整後的圖片

## 常見問題

1. **找不到模型？**
   - 確認模型檔案已正確放置在 model 目錄
   - 檢查模型格式是否支援
   - 第一次運行時會自動下載預設模型

2. **LoRA 沒有效果？**
   - 確認 LoRA 模型與基礎模型相容
   - 調整 LoRA 權重試試看
   - 檢查是否正確觸發 LoRA 的特定風格

3. **顯存不足？**
   - 降低圖片生成尺寸
   - 減少同時使用的 LoRA 數量
   - 啟用記憶體優化選項

