# 💊 경구약제 이미지 객체 검출

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8m-00A67D?style=flat-square"/>
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

<p align="center">
  <b>코드잇 부트캠프 초급 프로젝트 · 팀 프로젝트</b><br/>
  알약 사진 한 장에서 최대 4개의 약제를 찾아 위치와 종류를 맞히는 73클래스 객체 검출
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/10YRtRRwaORnLEYSEgRa3amK1wvelAG6Q#scrollTo=OHmaR0SA5o9c">📄 보고서 (Colab)</a> ·
  <a href="https://docs.google.com/presentation/d/1it5B1HVkXf2Zfq6S1dIB4PEeXuRfl3eq/edit?slide=id.p1#slide=id.p1">📊 발표자료</a> ·
  <a href="https://docs.google.com/spreadsheets/d/1O0zfyg_hiWu-L02N85qwJBho2GCq3n_pO2jacypseG8/edit?gid=1590775431#gid=1590775431">📋 협업일지</a>
</p>

---

## 📌 한눈에 보기

| 항목 | 내용 |
|------|------|
| 과제 | 경구약제 이미지 객체 검출 (Kaggle Private Competition, mAP 기반 채점) |
| 모델 | YOLOv8m (COCO 사전학습 가중치) |
| 클래스 | 73개 |
| 데이터 | Train 1,182장 / Val 307장 — Multi-label Stratified Split |
| 해상도 | 976×1280 (전 이미지 동일) |
| **최고 Kaggle 점수** | **0.99109** (v3 — 원본 해상도, 증강 없음) |

> 수치는 모두 팀 최종 노트북 [`oral_medication_final.ipynb`](./oral_medication_final.ipynb) §5–§6에 기록된 값입니다.

---

## 🗂️ 파일 구조

```
Oral-Medication-Project/
│
├── oral_medication_final.ipynb   # ★ 팀 최종 노트북 — EDA부터 버전별 비교·결론까지
│
├── notebooks/                    # 단계별 작업 노트북
│   ├── oral_medication.ipynb     #   데이터 확인 · 전처리 · 오버샘플링
│   └── YOLOv8m_best.ipynb        #   YOLOv8m 학습 설정 (v3 계열)
│
├── reports/
│   └── 20260330_tomas_yolo8m_v3.pdf   # YOLOv8m v3 실험 문서
│
├── LICENSE
└── README.md
```

★ 표시가 프로젝트의 최종 결과물입니다. 처음 보신다면 이 노트북부터 여시면 됩니다.

---

## 🔄 파이프라인

```
[원본 COCO JSON]
      │
      ▼
[전처리]          비정상 BBox 제거 (ann_id 903, 1157) · category_id 0~72 재매핑
      │
      ▼
[분할]            Multi-label Stratified Split — 73개 클래스가 Train·Val 양쪽에 모두 포함
      │
      ▼
[오버샘플링]      annotation 62개 미만 클래스(51개)를 이미지 복사로 보충 → Train 1,182 → 2,547장
      │
      ▼
[오프라인 증강]   복사된 이미지에만 회전·이동·스케일 ±3%
      │
      ▼
[YOLO 포맷 변환]  COCO [x, y, w, h] → YOLO [cx, cy, w, h] (정규화)
      │
      ▼
[YOLOv8m 학습]    imgsz 1280 · rect · AdamW · mosaic 0 · patience 20
      │
      ▼
[추론 → submission.csv]
```

**분할을 층화로 한 이유** — 일반 무작위 분할에서는 일부 클래스가 Val에만 배정됩니다. 한 이미지에 여러 약제가 함께 나오기 때문에 `iterative-stratification`으로 다중 라벨 층화 분할을 했습니다.

---

## 📊 실험 결과

### 버전별 성능

| 버전 | 주요 변경 | Val mAP50 | Best Epoch | Kaggle 점수 |
|------|---------|:---------:|:---------:|:----------:|
| v3 | 원본 해상도, 증강 없음 | 0.8591 | 50 | **0.99109** |
| v4 | 오버샘플링 단독 | 0.8712 | 20 | 0.98288 |
| v5 | 오버샘플링 + 온라인 증강 | 0.8670 | 32 | 0.92666 |
| v6 | 오프라인 증강 (HSV + 회전 10°) | 0.8688 | 6 | 미제출 |
| v7 | 오프라인 증강 (회전 3%) | 0.8619 | 29 | 0.97857 |
| tm_final | v7 파이프라인 + 팀원 설정 | 0.8699 | 31 | 0.91987 |

**Val mAP50이 높다고 Kaggle 점수가 높지 않았습니다.** Val mAP50이 가장 높은 v4·tm_final이 아니라, Val이 가장 낮은 v3가 Kaggle에서 가장 높았습니다.

### 소수 클래스 AP50

| class_id | 클래스 | annotation 수 | v3 | v4 | v6 | v7 |
|:-------:|-------|:----:|:----:|:----:|:----:|:----:|
| 65 | 브린텔릭스정 20mg | 7 | 0.2487 | 0.3317 | 0.2487 | 0.3317 |
| 60 | 아질렉트정 | 23 | 0.6093 | 0.8617 | 0.6219 | **0.8950** |
| 62 | 글리아타민연질캡슐 | 18 | 0.6225 | **0.9117** | 0.7450 | 0.7230 |
| 8 | 에어탈정 | 29 | 0.6251 | 0.8465 | **0.9310** | 0.7883 |
| 71 | 카발린캡슐 25mg | 23 | **0.8950** | **0.8950** | 0.8617 | 0.5983 |

### 추론 결과 비교

| 항목 | v3 | v4 | v7 | tm_final |
|------|:---:|:---:|:---:|:---:|
| 총 탐지 수 | 3,210 | 3,195 | 3,160 | 2,985 |
| 이미지당 평균 | 3.81 | 3.79 | 3.75 | 3.54 |
| 평균 confidence | 0.812 | 0.795 | 0.807 | **0.849** |

---

## 💡 배운 점

1. **촬영 환경이 균일하면 증강이 오히려 해가 된다** — 해상도·배경·조명이 모두 같은 데이터에서는 증강을 더할수록 Kaggle 점수가 떨어졌습니다. 원본을 그대로 학습한 v3가 최고점입니다.
2. **오버샘플링은 소수 클래스에 효과가 있다** — v4에서 취약 클래스 AP50이 평균 +0.15 올랐습니다. 다만 Val 향상이 Kaggle 점수로 이어지지는 않았습니다.
3. **데이터가 너무 적은 클래스는 기법으로 못 살린다** — annotation 7개인 65번 클래스는 어떤 버전에서도 0.34를 넘지 못했습니다.
4. **Val 점수만 보면 안 된다** — tm_final은 Val mAP50과 평균 confidence가 높았지만 총 탐지 수가 가장 적었고, 탐지 누락이 Kaggle 점수를 낮춘 것으로 분석했습니다.

### 목적별로 고른다면

| 목적 | 버전 | 근거 |
|------|:---:|------|
| Kaggle 점수 | v3 | 0.99109 |
| 소수 클래스 균형 | v4 | 취약 클래스 AP50 평균 +0.15 |
| 학습 안정성 | tm_final | 50 epoch 완주, early stopping 미발동 |
| 실환경 대응 | v7 | 경량 오프라인 증강으로 다양성 확보 |

---

## 🙋 담당 범위

부트캠프 팀 프로젝트로 진행했고, 이 저장소의 `notebooks/` 두 파일이 제가 맡은 부분입니다.

- **전처리와 오버샘플링** (`notebooks/oral_medication.ipynb`) — 비정상 BBox 제거, 다중 라벨 층화 분할, `TARGET_COUNT=62` 기준 오버샘플링, COCO→YOLO 변환
- **YOLOv8m 학습** (`notebooks/YOLOv8m_best.ipynb`) — 원본 해상도(`imgsz=1280`, `rect=True`)를 유지하는 학습 설정

팀 최종 노트북(`oral_medication_final.ipynb`)은 팀의 실험을 한데 정리한 최종본이며, 제 단독 작업이 아닙니다.

---

## 🚀 실행 환경

- **Google Colab** (GPU) 기준으로 작성됐습니다
- 데이터는 Google Drive에 두고 마운트해서 씁니다 — 노트북의 경로 변수(`/content/drive/MyDrive/...`)를 본인 위치로 바꿔 주세요
- 대회 데이터는 이 저장소에 포함돼 있지 않습니다

```bash
pip install ultralytics iterative-stratification opencv-python pandas matplotlib
```

---

## 🔧 기술 스택

| 영역 | 기술 |
|------|------|
| 모델 | Ultralytics YOLOv8m |
| 프레임워크 | PyTorch · TorchVision |
| 데이터 분할 | iterative-stratification (`MultilabelStratifiedShuffleSplit`) |
| 데이터 처리 | pandas · NumPy · OpenCV · Pillow |
| 시각화 | Matplotlib |
| 환경 | Google Colab · Google Drive |

---

<p align="center">
  <sub>MIT License · 자세한 내용은 <a href="./LICENSE">LICENSE</a></sub>
</p>
