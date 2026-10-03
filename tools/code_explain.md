### 8-4. 📖 코드 쉽게 읽기 (한 줄씩 간단 설명)

#### ① 00번: 모델 받기와 확인
```python
command = ["hf", "download", MODEL_ID, *MODEL_FILES, "--revision", MODEL_REVISION, "--local-dir", MODEL_DIR]
subprocess.run(command, check=True)          # 파이썬에서 터미널 명령 hf download를 실행
hashlib.sha256(path.read_bytes()).hexdigest()  # 받은 파일의 지문(SHA-256)을 계산
if 지문 != EXPECTED_MODEL_SHA256[파일]: raise ValueError  # 원본과 다르면 즉시 중단
```
- **한 줄 요약:** 정해진 버전의 모델 파일 3개를 받고, 지문이 원본과 같은지 확인한다.

#### ② 00번: 데이터 나누기
```python
rng = np.random.default_rng(SEED)              # seed 42로 고정 → 누가 돌려도 같은 결과
shuffled = rng.permutation(후보들)              # 한 클래스의 이미지 순서를 섞고
train += shuffled[:100]; val += shuffled[100:120]   # 앞 100장은 학습, 다음 20장은 검증
test = 원본 테스트에서 클래스마다 40장           # 테스트는 학습에 안 쓴 별도 이미지
np.savez_compressed(...)                       # 이미지·라벨을 .npz 파일로 저장
```
- **한 줄 요약:** CIFAR-100에서 5개 클래스만 골라 학습 500 / 검증 100 / 테스트 200장으로 나눈다.

#### ③ 01번: 모델 불러오기
```python
processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)  # 전처리 규칙 읽기
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
```
- `Auto...` 클래스는 `config.json`을 보고 **알맞은 모델 종류(여기선 ViT/DeiT)를 자동으로 고른다.**
- `local_files_only=True`: 인터넷에서 다시 받지 말고, 업로드한 파일만 쓰라는 뜻.
- `.to(device)`: 모델을 GPU(또는 CPU) 메모리로 옮긴다.

#### ④ 추론 함수 두 개
| 코드 | 쉬운 설명 |
|---|---|
| `processor(images=image, return_tensors="pt")` | 32×32 사진을 224×224로 키우고 숫자를 정규화해 **텐서**로 바꿈 |
| `.to(device)` | 입력도 모델과 같은 장치(GPU)로 이동 |
| `model.eval()` | 평가 모드: 학습용 동작(드롭아웃 등)을 끔 |
| `torch.inference_mode()` | 기울기 계산을 끔 → 빠르고 메모리 절약 |
| `model(**inputs).logits` | 클래스별 **원시 점수** (ImageNet이면 1,000개) |
| `.softmax(dim=-1)` | 점수를 **합이 1인 확률**로 변환 |
| `probabilities.topk(k)` | 확률이 큰 순서로 **k개**의 값과 번호를 꺼냄 |
| `id2label[번호]` | 번호를 사람이 읽는 이름(`cup` 등)으로 바꿈 |

#### ⑤ 학습 함수 두 개
| 코드 | 쉬운 설명 |
|---|---|
| `model.train()` | 학습 모드로 전환 |
| `optimizer.zero_grad()` | 지난 배치의 기울기를 **지움** (안 지우면 누적됨) |
| `logits = model(pixel_values=pixels).logits` | 16장 묶음(batch)을 넣어 예측 |
| `loss = criterion(logits, labels)` | 정답과 비교해 **얼마나 틀렸는지(손실)** 계산 (CrossEntropy) |
| `loss.backward()` | **역전파**: 각 가중치가 손실에 얼마나 책임 있는지(기울기) 계산 |
| `optimizer.step()` | 기울기 반대 방향으로 가중치를 **조금 수정** (Adam) |
| `requires_grad_(False)` | 이 가중치는 **얼림**(학습 안 함) |
| `requires_grad_(True)` | 이 가중치는 **녹임**(학습 대상) |

#### ⑥ 학습 루프 전체 흐름 (02번)
```python
for epoch in range(3):                   # 전체 학습 데이터를 3번 반복
    for pixels, labels in train_loader:  # 16장씩 꺼내서
        train_one_batch(...)             # 예측 → 손실 → 역전파 → 갱신
    validation = evaluate(val_loader)    # 한 바퀴 끝날 때마다 검증 정확도 측정
    if 지금이 최고 검증 정확도: best = 현재 가중치 복사   # 가장 좋았던 순간을 저장
model ← best                             # 마지막이 아니라 "가장 좋았던" 가중치로 되돌림
test = evaluate(test_loader)             # 마지막에 딱 한 번 테스트
model.save_pretrained(..., safe_serialization=True)   # .safetensors로 저장 → 다시 불러와 결과 같은지 확인
```
- **한 줄 요약:** 조금씩 고치고, 매번 검증하고, 가장 좋았던 모델을 골라 테스트한다.
