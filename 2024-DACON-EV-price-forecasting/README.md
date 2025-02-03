# 전기차 가격 예측 해커톤 : 데이터로 EV를 읽다!

## [DACON](https://dacon.io/competitions/official/236424/overview/description)

## [기간]
2024.12.02 ~ 2025.01.31

## [배경]
안녕하세요 데이커 여러분 :) 

전기차 가격 예측 해커톤: 데이터로 EV를 읽다!에 오신 것을 환영합니다! 

전기차 가격 예측은 빠르게 성장하는 전기차 시장에서 소비자와 제조사 모두에게 중요한 가치를 제공합니다.

정확한 가격 예측은 시장 경쟁력 분석, 소비자 구매 의사 결정 지원, 그리고 생산 및 유통 최적화에 기여할 수 있습니다.

이번 해커톤에서 전기차의 다양한 데이터를 바탕으로 가격을 예측하는 AI 알고리즘을 개발하는 것을 목표로 합니다.

여러분의 창의적인 아이디어와 데이터 분석 역량을 통해, 전기차 시장의 미래를 만들어 보세요!

## [주제]
전기차와 관련된 데이터를 활용하여 전기차 가격을 예측하는 AI 알고리즘 개발

## [설명]
전기차와 관련된 데이터를 활용하여 전기차 가격을 예측하는 AI 알고리즘을 개발해보세요!

## [My Solution]
- xgboost, lightgbm, random foreset, lstm, gru 구현 및 실험
    - 단일 모델로는 lightgbm의 성능이 가장 좋음
- 배터리용량 결측치 대체
    ```
    train['BatteryCapacity'] = train['BatteryCapacity'].fillna(train.groupby(['Manufacturer', 'Model'])['BatteryCapacity'].transform('mean'))
    test['BatteryCapacity'] = test['BatteryCapacity'].fillna(test.groupby(['Manufacturer', 'Model'])['BatteryCapacity'].transform('mean'))
    ```
- 파생 변수 생성
- category feature 중 제조사, 모델 feature에 target encoder 적용
- 나머지 category feature에 label encoder 적용
- kfold 10 적용
- optuna를 사용하여 최적 파라미터 탐색

## [Result]
- LEADERBOARD PRIVATE 13th (13 / 1276) / SCORE 1.20597