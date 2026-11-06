# 메모리 기반 협업 필터링 수정본

제안서와 팀의 MF·하이브리드 비교에 맞춰 평가를 통일하고, 사용자·상품 평균 보정만 사용하는 대조 모델을 추가한 버전입니다. 실제 재실행 결과는 `results/results.json`, 해석은 `검토결과.md`에 있습니다.

## 실행

Python 3.12에서 원본 데이터의 Software.csv와 Video_Games.csv를 이 폴더의 data 폴더에 넣습니다. 상품 메타데이터는 이번 실험에 필요하지 않습니다. 데이터와 가상환경은 배포 ZIP에 포함하지 않습니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe test_evaluation.py
.\.venv\Scripts\python.exe run_reviewed_memory.py
```

다른 위치의 데이터를 쓰려면 `--data-dir "데이터 폴더 경로"`를 지정합니다. 결과 경로는 `--output-dir "결과 폴더 경로"`로 바꿀 수 있습니다. 기존 결과를 보존하려면 다른 출력 폴더를 지정하세요.

## 수정 사항

- RMSE와 F1을 동일한 예상 평점으로 계산하는 `Improved rating`을 주 비교 모델로 제공합니다.
- 이웃 투표 기반 추천은 `Improved vote`라는 별도 방법으로 보고합니다. 평점 예측값이 아니므로 이 행의 RMSE는 null입니다.
- `Bias only`는 개선 모델과 완전히 동일한 학습 평균·사용자 편향·상품 편향을 사용하되 이웃 잔차만 제외합니다. 이웃 잔차의 추가 기여를 확인하는 비교입니다.
- 기본 F1은 전체 평가 사용자 평균입니다. 관련 테스트 상품이 없는 사용자는 0으로 포함합니다. 기존 보고서와 비교할 관련 상품 보유 사용자 평균도 별도로 기록합니다.
- Top-10 동점은 인코딩된 상품 인덱스가 작은 순으로 처리합니다. 이미 본 상품과 다른 도메인 상품은 추천에서 제외하며, 후보가 부족하면 제외된 상품으로 채우지 않습니다.
- 기존 kNN, 전체 평균, 인기 추천도 같은 평가기로 다시 실행했습니다. 기존 kNN의 k=20은 과거 검증에서 정한 값을 유지한 비교입니다.
- 추천 방식 변경과 평균 대상 변경, 동점 처리 변경이 있으므로 과거 JSON의 수치를 새 표에 섞지 않습니다.
- 유사도 계산을 32명 단위로 나누어 팀 코드에서 큰 사용자 묶음을 전달할 때의 순간 메모리 사용을 줄였습니다. 반환 점수 배열 자체는 사용자 수×상품 수이므로 평가도 작은 묶음을 사용하세요.

## 팀 평가 규칙

| 항목 | 규칙 |
|---|---|
| 데이터 분할 | 기존 `recsys.time_split`: 사용자별 시간순, 검증 약 10%, 테스트 약 20%, 각각 최소 1개 |
| 학습 | train만 사용. 검증 데이터는 설정 선택과 추천 제외 목록에 사용 |
| 평가 표본 | 분야별 최대 10,000명, 시드 42 |
| 관련 상품 | 테스트 평점 4점 이상 |
| 추천 개수 | Top-10, Precision 분모는 10 |
| 기본 평균 | 모든 평가 사용자. 관련 상품이 없으면 P/R/F1=0 |
| 보조 평균 | 관련 테스트 상품이 있는 사용자만 |
| RMSE | 평가 표본의 모든 테스트 평점에 대한 오차. 사용자별 RMSE 평균이 아님 |
| F1 | 사용자별 F1을 계산한 뒤 평균. 평균 Precision·Recall의 조화평균이 아님 |
| 설정 선택 | Software 검증 표본 2,000명, 시드 142, 투표 추천의 기본 F1 최대, 동률이면 평점 RMSE 최소 |
| 게임 내부 | Software 설정 그대로 사용, 게임 train으로 재학습 |
| 크로스 도메인 | 공통 사용자 중 20%의 게임 기록 전체 숨김. Software만 유사도에 사용 |

평점 기반 변형도 기존 개선안과 동일한 설정을 공유합니다. 평점 기반 F1이나 RMSE를 별도로 최적화한 결과는 아닙니다. 크로스 도메인은 사용자 홀드아웃이며 미래 시점 예측 실험이라고 해석하지 않습니다.

## MF 및 하이브리드 담당자 연결

같은 Encoder, 데이터 분할, 평가 사용자, 후보 마스크를 사용해야 합니다. 기존 `base.evaluate` 대신 아래 공통 평가기를 사용하면 평균 사용자 기준과 동점 처리가 통일됩니다.

```python
from improved_memory import ImprovedUserKNN
from evaluation import evaluate_model

knn = ImprovedUserKNN(k=2000, popularity_power=0.0, shrink=5.0)
knn.fit(u, i, r, enc.shape)
knn_result = evaluate_model(knn.scores, users, known, test)
mf_result = evaluate_model(mf.scores, users, known, test)

# alpha는 검증 데이터에서 선택하고 테스트 전에 고정
hybrid = base.Hybrid(mf, knn, alpha)
hybrid_result = evaluate_model(hybrid.scores, users, known, test)
```

`knn.scores()`는 1~5 예상 평점으로 MF 평점과 결합할 수 있습니다. `ranking_scores()`는 이웃 투표 값이므로 MF 평점과 직접 가중 평균하지 마세요. 평점 예측용 alpha와 추천용 alpha를 따로 선택한다면 두 변형을 별도 행으로 보고하세요.

크로스 도메인에는 다음과 같이 Software 유사도 자료와 게임 기록이 있는 이웃 마스크를 전달합니다. 변수 구성은 `run_reviewed_memory.py`의 실제 실행 부분을 참고하세요.

```python
knn.fit(*enc.arrays(joint_train), enc.shape,
        similarity_data=enc.arrays(software), neighbor_mask=has_target)
metrics = evaluate_model(knn.scores, test_users, no_target_history, target_test,
                         candidates=target_items)
```

이 수정본은 MF·하이브리드를 학습하거나 그 성능을 검증하지 않습니다. 기존 `recsys.py`는 참조 및 원본 모델 실행용으로 보존했습니다. `python recsys.py`를 실행하면 이전 전체 실험/이전 평가 규칙을 사용하므로 수정본의 실행 명령과 혼동하지 마세요. 이전의 `evaluate_pair`는 서로 다른 점수의 지표를 한 행에 합치므로 제거했습니다.

## 해석의 한계

이전에 확인한 테스트 세트를 재사용한 개선 실험입니다. 코드가 검증 세트로 설정을 선택하더라도 새롭고 독립적인 테스트에서 검증됐다고 주장할 수 없습니다. 신뢰구간이나 다중 시드 검증도 수행하지 않았으므로 작은 차이를 유의한 차이라고 부르지 않습니다.

평균 보정 제거 비교 외에 긍정 피드백 유사도, k 확대, 투표 순위, 인기도 대체의 개별 효과는 분리하지 않았습니다. 인기도 대체를 포함한 이웃 기반 방법이며 잠재 요인 MF를 사용하지 않습니다.

## 파일

- `improved_memory.py`: 개선 kNN과 평균 보정 대조 모델
- `evaluation.py`: 팀 공통 평가 및 동점 규칙
- `run_reviewed_memory.py`: 검증 설정 선택, 세 실험과 대조 모델 재실행
- `test_evaluation.py`: 평가·마스킹·크로스 도메인 유사도 검증 5개
- `results/validation_selection.json`: 테스트 평가 전에 저장하는 설정 선택 기록
- `results/results.json`: 실제 재실행 결과와 입력 데이터 SHA-256
- `comparison.png`: 기본 평균 기준 결과 그림
- `검토결과.md`: 실제 수치에 따른 해석과 발표 문장
