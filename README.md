# Retrieval Module For Open-set Labeling

## TL;DL
- 레이블링을 좀더 쉽게 하기 위해서, 지금 DB에 쌓여있는 이미지들(query set) 중에서 사람이 정해준 샘플 이미지(support set)과 제일 비슷해 보이는 이미지들의 순위를 매겨주는 Retriever 입니다
- 그 중에서도, 현재 대분류 모델의 08 class, 16 class에 대한 데이터를 가져오는 코드를 작성했습니다

## Code Strudcture
```python
retriever
├── downstream_modules   # util functions 
│   ├── data_utils.py       
│   ├── train_utils.py      
│   └── utils.py             
├── model                # model architecture 관련 (ResNet, PMG)
├── ...
│── config.py            # pretrained model load, dataset setting 관련
│── dataset.py           # support_set, query_set 로드 관련
│── utils.py             # misc
└── retriever.py         # retrieve를 하기 위한 main 함수 

── data_folder           # pre-defined dataset (직접 데이터 추가필요)
── -.pth                 # pretrained weight   (직접 pretrained weight 추가필요)
```

## Environment Setting
```shell
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install easydict
```

## Input Dataset
- (1) Query Dataset from bigquery
```shell
SELECT *
FROM `socar-data.socar_data_ml.car_state_classifier_inference_result`
WHERE 1=1
AND uploader_type = 'MEMBER'
AND DATE(created_at, 'Asia/Seoul') BETWEEN "2022-05-01" AND "2022-11-25" 

UNION ALL

SELECT *
FROM `socar-data.socar_data_ml.car_state_classifier_inference_result_2021_2022_mid` 
WHERE 1=1
AND uploader_type = 'MEMBER'
AND DATE(created_at, 'Asia/Seoul') BETWEEN "2021-01-01" AND "2022-04-30"
```
- (2) Support set (08, 16 class) from bucket 
```shell 
gsutil cp gs://socar-data-temp/esther/car_state_classifier/support_set_candidates/08_inner_cupholder_dirt.zip .
gsutil cp gs://socar-data-temp/esther/car_state_classifier/support_set_candidates/16_inner_seat_dirt.zip .
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
python retriever.py 
````

