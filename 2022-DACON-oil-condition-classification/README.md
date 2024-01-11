# DACON 건설기계 오일 상태 분류 AI 경진대회

## [DACON](https://dacon.io/competitions/official/236013/overview/description)

#### [배경]
건설기계 분야의 데이터를 분석, 활용하는 방안을 제시하는 경진대회를 개최하여 참신한 아이디어를 발굴하고 해당 분야의 인공지능 기술의 발전을 도모하려 합니다.
또한 건설 장비 산업의 지능화에 대한 연구활동 홍보 및 우수 인재를 발굴하고자 합니다.

#### [주제]
건설장비에서 작동오일의 상태를 실시간으로 모니터링하기 위한 오일 상태 판단 모델 개발 (정상, 이상의 이진분류)

#### [설명]
건설 장비 내부 기계 부품의 마모 상태 및 윤활 성능을 오일 데이터 분석을 통해 확인하고, AI를 활용한 분류 모델 개발을 통해 적절한 교체 주기를 파악하고자 합니다.
이번 경진 대회에서는 모델 학습시에는 주어진 모든 feature를 사용할 수 있으나, 진단 테스트시에는 제한된 일부 feature만 사용 가능합니다.
따라서 진단 환경에서 제한된 feature 만으로도 작동 오일의 상태를 분류할 수 있는 최적의 알고리즘을 만들어주세요.

#### [주최 / 주관]
주최 : 현대제뉴인 [링크]
후원 : AWS
주관 : 데이콘

## Development Environmnet
- Ubuntu 18.04
- GTX 3090 1EA
- Jupyter Notebook


## Solution
- Teacher : XGBClassifier, LGBMClassifier, CatBoostClassifier Ensemble (Soft Voting)
- Student : XGBRegressor, LGBMRegressor, CatBoostRegressor Ensemble (Soft Voting)
- Find Best Parameters : Optuna
- 영향력 적은 20개의 Columns 제거
