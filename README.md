# Retrieval Module For Open-set Labeling

## TL;DL
- support_set (=사람이 지정해주는 샘플 데이터)를 사용하여 전체 query_set(=from open-set DB) 중 support_set과 같은 클래스로 간주되는 것에 대한 우선순위를 매긴다
- support_set의 갯수는 batch_size에 의해서 결정되며, 머신에 따라 size 조정이 가능

## Code Strudcture
```python
retriever
├── downstream_modules   # util functions 
│   ├── data_utils.py       
│   ├── train_utils.py      
│   └── utils.py             
├── model                # model architecture 관련 (ResNet, PMG)
├── ...
│── result               # retrieved results
│   ├── ...
│── config.py            # pretrained model load, dataset setting 관련
│── dataset.py           # support_set, query_set 로드 관련
│── utils.py             # misc
└── main.py              # 학습을 위한 main 함수 

── data_folder           # pre-defined dataset (직접 데이터 추가필요)
── -.pth                 # pretrained weight   (직접 pretrained weight 추가필요)
```

## Environment Setting
```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install easydict
```

## Download pre-trained model weights
```python
# sofar v3 best model 
gstuil cp gs://socar-data-temp/tigger/artifacts/car_state_classifier/sofar_v3_best_model/imagenet=ce_sofarv3=byol_finetune_best_model.pth path_to_save

# (USE THIS) sofar v3 + calibration best model 
gsutil cp gs://socar-data-temp/tigger/artifacts/car_state_classifier/sofar_v3_best_model/calibrated_lb_smooth=0.05/lb_smooth=0.05_best_model.pth path_to_save
```

## How To Retrieve
```python
python main.py 
````

