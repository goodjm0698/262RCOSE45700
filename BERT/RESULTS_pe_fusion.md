# BERT-base NPU 측정 결과

실행 ID `20261119T032413Z-616241`. 원본은 `BERT/vitstyle/runs/20261119T032413Z-616241/`에 있음.

## 실험 조건

- 모델: `textattack/bert-base-uncased-SST-2`, BF16, 길이 128, `cpu_fallback=False`, attention 구현 `sdpa`
- 마스크: `create_bidirectional_mask`를 passthrough로 패치하고 4D additive mask를 CPU에서 만들어 NPU로 넘김 (step4 방식)
- 입력: step4의 실제 문장을 128로 패딩함. batch B는 1~B번 문장, 16개를 넘으면 16문장을 순서대로 반복해 채움 (batch [1, 16, 64]), id는 int32
- PE: worker마다 장치 사용 전에 `furiosa.torch.set_fusion(4)`을 한 번 호출하고 `furiosa:0` 사용. 실제 적용값은 `(Fusion(num_fusion=4, num_pe_per_cluster=4, num_chips=1, num_clusters_per_chip=1))`
- 측정 모드: ['compile']만 측정함. 다른 모드의 칸은 비어 있음
- 한 번 실행할 때마다 첫 호출 1회 + 워밍업 10회 + 반복 50회. 조건마다 새 프로세스로 1세트 실행함. 실행 순서는 회차마다 바꿈
- 대표값: 반복 50회(세트 1개 × 50회)를 모두 모은 평균 ± 표준편차. 중앙값과 세트별 평균도 같이 적음. 정상 종료한 세트만 포함함
- 첫 호출은 성능 비교에서 빼고 기록만 함. 첫 호출·워밍업 합·A는 세트 1개의 중앙값 (min~max)
- 캐시: 실행마다 `runs/<id>/caches/<stage>/<case>`를 새로 만들어 씀. 팀 공용 캐시는 사용하지 않음
- 타이머: `stream.synchronize()` → `perf_counter_ns` → `runner(**inputs)` → `stream.synchronize()`. 입력 전송과 출력 CPU 복사는 타이머 밖임
- A = runner 준비(S) + 첫 호출(F) + 워밍업 합(W). 모델 로드와 NPU 전송 시간은 제외함. T(N) = A + N × 반복 평균
- NPU 사용 시간 합계 988초 (예산 14400초)

## 표 1. 반복 시간 (ms, 대표값은 50회 전체 평균 ± 표준편차)

| batch | 모드 | 평균 ± 표준편차 | 중앙값 | n | 세트1 평균 |
|---|---|---|---|---|---|
| 1 | eager | — | — | 0 | — |
| 1 | compile | 1.388 ± 0.012 | 1.387 | 50 | 1.388 |
| 16 | eager | — | — | 0 | — |
| 16 | compile | 7.826 ± 0.012 | 7.827 | 50 | 7.826 |
| 64 | eager | — | — | 0 | — |
| 64 | compile | 22.326 ± 0.048 | 22.319 | 50 | 22.326 |

## 표 1-2. 첫 호출·워밍업·A (기록용, 세트 1개의 중앙값, 괄호는 min~max)

| batch | 모드 | 첫 호출 | 워밍업 10회 합 (ms) | A |
|---|---|---|---|---|
| 1 | eager | — s | — | — s |
| 1 | compile | 186.02 (186.02~186.02) s | 14.45 (14.45~14.45) | 186.28 (186.28~186.28) s |
| 16 | eager | — s | — | — s |
| 16 | compile | 327.87 (327.87~327.87) s | 78.91 (78.91~78.91) | 328.19 (328.19~328.19) s |
| 64 | eager | — s | — | — s |
| 64 | compile | 452.10 (452.10~452.10) s | 224.36 (224.36~224.36) | 452.57 (452.57~452.57) s |

## 표 2. 배속 (eager 반복 평균 / compile 반복 평균)

| batch | 대표 (전체 평균끼리) | 세트1 |
|---|---|---|
| 1 | — | — |
| 16 | — | — |
| 64 | — | — |

대표 배속은 전체 평균끼리 나눈 값임. 세트별 값은 같은 세트의 eager와 compile 평균끼리 나눈 값임.

## 표 3. 손익분기점

| batch | 예측 N_win 중앙값 (min~max) | 실측 교차 N | 교차 후 우위 | 상태 |
|---|---|---|---|---|
| 1 | — | — | — | 미실행 |
| 16 | — | — | — | 미실행 |
| 64 | — | — | — | 미실행 |

이번 실행에서는 cumulative 단계를 돌리지 않았음. 실측 칸은 비어 있고 예측값만 있음.

N은 첫 호출과 워밍업 10회를 마친 뒤 추가로 호출한 횟수임. 실측은 별도 프로세스와 새 캐시에서 대표 회차(N_win이 중앙값인 회차)의 예측값과 비교함.

## 표 4. 출력 비교 (eager vs compile, BF16 logits를 CPU FP32로 바꿔 비교)

| 입력 | batch | 최대 절대오차 | 평균 절대오차 | cosine 평균 | cosine 최소 | Top-1 일치 |
|---|---|---|---|---|---|---|
| step4 문장 전체 (출력 비교 worker) | 1 | — | — | — | — | — |
| step4 문장 전체 (출력 비교 worker) | 16 | — | — | — | — | — |
| step4 문장 전체 (출력 비교 worker) | 64 | — | — | — | — | — |
| 성능 측정 출력 (마지막 반복, 1회 중 최악값) | 1 | — | — | — | — | — |
| 성능 측정 출력 (마지막 반복, 1회 중 최악값) | 16 | — | — | — | — | — |
| 성능 측정 출력 (마지막 반복, 1회 중 최악값) | 64 | — | — | — | — | — |

허용 오차 기준은 아직 정하지 않았으므로 합격 판정은 하지 않음(pending). BERT logits는 2차원이라 cosine이 높게 나오기 쉬움.

## 표 6. PE fusion 비교 (compile 반복 시간)

| batch | fusion 1 compile 평균 ± SD (ms) | fusion 2 compile 평균 ± SD (ms) | 배속 (fusion 1 / fusion 2) | fusion 4 compile 평균 ± SD (ms) | 배속 (fusion 1 / fusion 4) |
|---|---|---|---|---|---|
| 1 | 2.029 ± 0.024 | 1.649 ± 0.013 | 1.230 | 1.388 ± 0.012 | 1.462 |
| 16 | 17.481 ± 0.021 | 10.821 ± 0.053 | 1.615 | 7.826 ± 0.012 | 2.234 |
| 64 | 74.441 ± 0.018 | 34.754 ± 0.012 | 2.142 | 22.326 ± 0.048 | 3.334 |

fusion n = `furiosa.torch.set_fusion(n)`으로 PE n개를 `furiosa:0` 하나로 묶은 것. fusion 1은 SDK 기본값(PE 1개)임. 배속은 같은 batch의 fusion 1 평균을 해당 fusion 평균으로 나눈 값. 평균 ± SD의 n은 run마다 다를 수 있음(세트 수 × 반복 수). 출처 run: `20261110T140237Z-61d998` (fusion 1 batch 1, fusion 1 batch 2, fusion 1 batch 4, fusion 1 batch 8, fusion 1 batch 16); `20261113T051434Z-fe0cc4` (fusion 1 batch 32, fusion 1 batch 64); `20261119T030722Z-d74488` (fusion 2 batch 1, fusion 2 batch 16, fusion 2 batch 64); `20261119T032413Z-616241` (fusion 4 batch 1, fusion 4 batch 16, fusion 4 batch 64). 다른 run은 읽기만 함.

