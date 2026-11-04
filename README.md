# Cross-Domain Product Recommendation

**Software → Video Games | 2026 Machine Learning Term Project · Team 4**

Amazon Reviews 2023의 Software 및 Video Games 평점을 이용해 사용자별 평점을 예측하고 상위 10개 상품을 추천합니다. 메모리 기반 협업 필터링(kNN), 모델 기반 협업 필터링(MF), 하이브리드 필터링을 비교하는 프로젝트입니다.

## 진행 상태

| 방법 | 상태 |
| --- | --- |
| 메모리 기반 협업 필터링 | Software·Video Games·크로스 도메인 학습 및 평가 완료 |
| 모델 기반 협업 필터링(MF) | 담당자 구현·학습·평가 예정 |
| 하이브리드 필터링 | 담당자 구현·학습·평가 예정 |


## 구조

```text
.
├── recsys.py                  # 공통 데이터 처리, 모델, 평가 함수
├── run_memory.py              # Software / Video Games 메모리 실험
├── run_memory_cross.py        # 메모리 기반 크로스 도메인 실험
├── requirements.txt
├── data/                      # 로컬 데이터 (Git 제외)
│   └── README.md
├── outputs/                   # 공유 평가 결과
│   ├── memory_software_full_10000/
│   ├── memory_video_games_full_10000/
│   └── memory_cross_domain/
└── docs/
    ├── MEMORY_RESULTS.md      # 메모리 실험 결과와 해석
    └── ML_Term_Project_Proposal_Team4.docx
```

모든 실험은 하나의 `recsys.py`를 사용합니다. 데이터와 결과도 루트의 `data/`, `outputs/`에서 관리합니다.

## 설치와 실행

메모리 담당자가 기록한 실행 환경은 Python 3.12.14입니다. 패키지 버전은 제공된 `requirements.txt`를 유지했습니다.

저장소 루트에서 가상환경을 만듭니다.

```bash
python -m venv .venv
```

Windows PowerShell에서는 활성화 없이 다음과 같이 설치·실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_memory.py --eval-users 100
```

먼저 [데이터 안내](data/README.md)에 따라 `data/`에 `Software.csv`, `Video_Games.csv`를 준비하세요. 위 명령은 평가 사용자 100명의 예비 실행이며 학습 데이터 전체는 유지합니다.

정식 평가는 Software부터 순서대로 실행합니다.

```powershell
.\.venv\Scripts\python.exe run_memory.py --eval-users 10000
.\.venv\Scripts\python.exe run_memory.py --category Video_Games --eval-users 10000
.\.venv\Scripts\python.exe run_memory_cross.py
```

macOS/Linux에서는 Python 경로를 `.venv/bin/python`으로 바꿉니다. 사용 중인 Python 환경에 패키지가 설치되어 있다면 `python`으로 실행할 수 있습니다.

게임과 크로스 도메인 실험은 `outputs/memory_software_full_10000/results.json`에서 선택된 k를 읽습니다. 실행하면 같은 경로의 결과가 갱신됩니다.

`python recsys.py`는 MF·하이브리드까지 포함한 전체 실험 출발 코드를 실행합니다. 현재 완료된 메모리 실험의 재현은 위 전용 실행 파일을 사용하세요.

## 실험 설계

1. **Software 내부:** 사용자별 시간순 분할로 학습하고, 검증 데이터에서 설정값을 선택합니다.
2. **Video Games 내부:** Software에서 선택한 설정값으로 게임 학습 데이터에서 다시 학습한 후 평가합니다.
3. **크로스 도메인:** 공통 사용자 일부의 게임 평점을 전부 숨기고, Software 평점과 다른 사용자의 게임 평점으로 숨긴 게임 취향을 예측합니다.


## 자료

원본 데이터, 가상환경, 캐시 및 대형 모델 파일은 Git에서 제외합니다. 집계 결과와 문서는 포함합니다.

- [프로젝트 제안서](docs/ML_Term_Project_Proposal_Team4.docx)
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
- [데이터셋 제공처](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
