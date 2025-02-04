### 피어리뷰 템플릿

- [x] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
  - 완성도 검토
    - [x] 데이터 전처리, 모델 학습, 추론까지 포함된 완성도 있는 코드 제출
    - [x] Seq2Seq 모델에 어탠션 매커니즘 적용해서 요약 성능을 개선하려는 시도
    - [x] summa 라이브러리를 활용한 추출적 요약
  - 개선할 점
    - 다양한 하이퍼파라미터 실험 결과 비교 그래프가 필요해 보임.
- [x] **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**

  - [x] 모델 선정 이유
    - Seq2Seq + Attention 모델을 선택한 이유에 대한 설명이 포함됨.
  - [ ] 하이퍼 파라미터 선정 이유
    - dropout, embedding_dim, learning_rate 등의 선택이 있지만, 설명이 부족함
  - [x] 데이터 전처리 이유 또는 방법 설명
    - HTML 태그 제거, 약어 정규화, 불용어 제거 등의 과정이 코드로 구현되었으며 설명도 비교적 잘 작성됨.

- [x] **3. 체크리스트에 해당하는 항목들을 수행하였나요? (문제 해결)**

  - [x] 데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)

  ```python
  # train/test 데이터 분리
  n_of_val = int(len(encoder_input) * 0.2)
  encoder_input_train = encoder_input[:-n_of_val]
  decoder_input_train = decoder_input[:-n_of_val]
  decoder_target_train = decoder_target[:-n_of_val]

  encoder_input_test = encoder_input[-n_of_val:]
  decoder_input_test = decoder_input[-n_of_val:]
  decoder_target_test = decoder_target[-n_of_val:]
  ```

  - [ ] 하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
  - [x] 각 실험을 시각화하여 비교하였나요?
    - loss와 val_loss의 시각화
    ```python
    # 모델의 손실 함수 시각화
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    ```
  - [ ] 모든 실험 결과가 기록되었나요?

- [x] **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**

  - [x] 배운 점
  - [x] 아쉬운 점
  - [x] 느낀 점
  - [x] 어려웠던 점

- [ ] **5. 앱으로 구현하였나요?** - [ ] 구현된 앱이 잘 동작한다. - [ ] 모델이 잘 동작한다.
