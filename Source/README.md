소스 파일 구성에 대한 설명
===
올린 소스에 대해서는 원본 데이터 파일과 라벨 파일이 없기 때문에 동작하지는 않습니다.
구성 파일에 대한 설명입니다.

- `TrainModel_for_final.py`: 모델을 정의하고, 훈련시키는 파일입니다. 훈련 과정마다 중간 결과를 볼 수 있도록 작성했습니다.
- `test_models_for_final.py`: 테스트 세트에 대하여 정확도를 계산하는 파일입니다.
- `make_graph.py`: 로그 파일을 읽어와 그래프로 표현하는 파일입니다.
- `load_data_t.py`: `TrainModel_for_final.py`와 `test_models_for_final.py`에서 사용하는 함수가 담긴 파일입니다. 데이터를 로드하고 `Augmentation`을 하는 역할을 합니다.
- `verify_coord.py`: 라벨이 올바르게 그려져 있는지 확인하는 파일입니다.
실행 하면 `./Verify` 폴더에 네 개의 라벨 점이 찍힌 이미지가 저장됩니다.
- `verify_data.py`: 라벨이 제한 조건을 만족하는지 검증하는 파일입니다.
