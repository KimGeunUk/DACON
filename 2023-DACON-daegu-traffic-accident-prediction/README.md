# 대구 교통사고 피해 예측 AI 경진대회

## [DACON](https://dacon.io/competitions/official/236193/overview/rules)

### [배경]
이동수단의 발달에 따라 다양한 유형의 교통사고들이 계속 발생하고 있습니다. 
한국자동차연구원과 대구디지털혁신진흥원에서는 해당 사고의 원인을 규명하고 사고율을 낮추기 위해, 
시공간 정보로부터 사고위험도(ECLO)를 예측하는 AI 알고리즘 발굴을 목표로 본 대회를 개최합니다. 

※ ECLO(Equivalent Casualty Loss Only) : 인명피해 심각도
ECLO = 사망자수 * 10 + 중상자수 * 5 + 경상자수 * 3 + 부상자수 * 1
본 대회에서는 사고의 위험도를 인명피해 심각도로 측정

### [주제]
시공간 정보로부터 사고위험도(ECLO) 예측 AI 모델 개발

### [설명]
사고 발생 시간, 공간 등의 정보를 활용하여 사고위험도(ECLO)를 예측하는 AI 알고리즘 개발

### [주최 / 주관 / 운영]
주최: 산업통상자원부, 대구광역시
주관: 한국자동차연구원, 대구디지털혁신진흥원
운영: 데이콘

## [My Solution]
- 제공되는 외부 데이터(보안등, 어린이 보호 구역, 주차장 정보, CCTV 정보)를 요약하여 train_new.csv 파일 생성
- 교통사고는 특정 지형(골목, 내리막길, ...)에 영향을 받음 -> 위도, 경도 정보 포함
- 교통사고는 특정 시간(새벽, 퇴근시간, 계절, ...)에 영향을 받음 -> 시간 정보, 계절 정보 포함
- [AutoML mljar-supervised](https://supervised.mljar.com/) 라이브러리 사용
- XGBoost, CatBoost, LightGBM Esemble
- 최종 제출 = 학습시킨 모델 중 상위 3개의 모델 Bagging

## [Result]
- LEADERBOARD PRIVATE 34th (34/941, 0.27%) / SCORE 0.42688