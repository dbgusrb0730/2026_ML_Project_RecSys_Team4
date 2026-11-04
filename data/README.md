# 데이터 준비

원본 데이터는 로컬에 보관하고 Git에는 업로드하지 않습니다. 이 폴더의 안내 파일만 버전 관리합니다.

## 파일 배치

```text
data/
├── README.md
├── Software.csv
├── Video_Games.csv
├── meta_Software.jsonl.gz
└── meta_Video_Games.jsonl.gz
```

평점 CSV에는 `user_id`, `parent_asin`, `rating`, `timestamp` 열이 필요합니다. 평점은 1~5, 시간은 밀리초 단위입니다.

| 데이터 | 평점 수 | 사용자 수 | 상품 수 |
| --- | ---: | ---: | ---: |
| Software | 1,276,840 | 146,396 | 17,591 |
| Video Games | 814,586 | 94,762 | 25,612 |

위 수치는 팀에서 사용하는 Amazon Reviews 2023 **5-core** 평점 파일 기준입니다. 같은 카테고리의 다른 버전이나 전처리된 파일은 같은 실험 결과를 보장하지 않습니다. 공통 사용자는 4,026명입니다.

## 어디서 준비하나요?

- 원본 제공처: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

메모리 기반 세 실험에는 두 CSV가 있으면 됩니다. 메타데이터는 베이스 코드에서 상품명을 표시하는 예시를 생성할 때 사용합니다.

루트의 `recsys.py`, `run_memory.py`, `run_memory_cross.py`는 이 폴더를 공통으로 사용합니다. 저장소 루트에서 실행하세요.
