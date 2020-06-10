# DRCD-ExerciseMaskedLM-BertForMaskedLM
使用[台達電的資料集](https://github.com/DRCKnowledgeTeam/DRCD)練習MaskLM的做法並fine-turing BertForMaskedLM

## 檔案說明
### Data
- preprocess_data.py : maskLM的前處理
- train.py : 模型訓練(BertForMaskedLM fine-tune)
- predict.py : maskLM的預測
- requestment.txt : 紀錄需要安裝的環境
## 環境需求
- python 3.6+
- pytorch 1.3+
- transformers 2.2+
- CUDA Version: 10.0
