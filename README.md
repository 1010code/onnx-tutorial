
### TensorFlow Serving
透過 TensorFlow Serving，開發者可以使用 TensorFlow 訓練模型，然後用 TensorFlow Serving 的 API 處理來自客戶端的數據輸入，類似Web API功能，訓練好的AI Model直接使用，不必再用Python框架--Flask/Django寫成Web API。

### ONNX Runtime
ONNX Runtime是用來執行ONNX模型的推理引擎，可以在不同作業系統和硬體上執行，現在開發者可以在智慧型手機、嵌入式裝置和物聯網等邊緣裝置，以ONNX Runtime執行機器學習模型。

## 分析
平常要最佳化一個機器學習模型以讓它適用於不同的硬體平台並不容易，開發人員必須針對每一個硬體平台與軟體配置手動調整模型。透過通用格式與 Runtime 概念，可減少調整機器學習模型以部署於不同平台的力氣，藉由自動最佳化TensorFlow、MXNet、PyTorch、ONNX與XGBoost等模型，在不喪失精確性的情況下，讓其執行速度達到原始模型的兩倍；此外，它還能將模型轉換為高效通用格式，以解決軟體相容性問題。

### Neo-AI
AWS 開源 Neo-AI 專案，提供訓練並輸出跨平台的機器學習模型。AWS 提供的 Amazon SageMaker 服務，能協助開發人員或資料科學家快速建立、訓練與部署機器學習模型。而 Amazon SageMaker Neo 則是 SageMaker 服務的一項功能，只要訓練一次就能以最佳化效能在雲端或其它硬體平台上運作。

## Reference
- [tf2onnx getting_started](https://github.com/onnx/tensorflow-onnx/blob/main/examples/getting_started.py)
- [onnxruntime not supported non-tensor inputs/outputs](https://github.com/microsoft/onnxruntime/issues/4294)
- [ONNX Runtime + FastAPI Deploying for Production Inference](https://blog.krudewig-online.de/2021/09/09/Machine-Learning-Models-Light-as-a-Feather.html)