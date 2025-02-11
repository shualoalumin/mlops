# 필요한 라이브러리 임포트
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import re
import urllib.request
import matplotlib.pyplot as plt
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu

# NLTK 데이터 다운로드
nltk.download('punkt')

# 2. 상수 및 하이퍼파라미터 정의
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1
EPOCHS = 20
BATCH_SIZE = 64
BUFFER_SIZE = 20000  # 데이터셋 크기보다 크게 설정되어 있어 충분합니다

# 데이터 다운로드 및 로드
def load_data():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", 
                              filename="ChatbotData.csv")
    train_data = pd.read_csv('ChatbotData.csv')
    print('데이터셋 크기 :', len(train_data))
    return train_data

# 문장 전처리 함수
def preprocess_sentence(sentence):
    sentence = sentence.strip()
    sentence = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

# 데이터 전처리 및 토크나이저 설정
def prepare_data(train_data):
    questions = [preprocess_sentence(sentence) for sentence in train_data['Q']]
    answers = [preprocess_sentence(sentence) for sentence in train_data['A']]
    
    # 텍스트 인코더 생성
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13)
    
    # 토큰 설정
    START_TOKEN = [tokenizer.vocab_size]
    END_TOKEN = [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2
    MAX_LENGTH = 40
    
    return questions, answers, tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE, MAX_LENGTH

# 데이터셋 생성
def create_dataset(questions, answers, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH):
    def tokenize_and_filter(inputs, outputs, max_length=40):
        tokenized_inputs, tokenized_outputs = [], []
        
        for (sentence1, sentence2) in zip(inputs, outputs):
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
            
            if len(sentence1) <= max_length and len(sentence2) <= max_length:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)
        
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=max_length, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=max_length, padding='post')
        
        return tokenized_inputs, tokenized_outputs

    questions_encoded, answers_encoded = tokenize_and_filter(questions, answers)
    
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000
    
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_encoded,
            'dec_inputs': answers_encoded[:, :-1]
        },
        {
            'outputs': answers_encoded[:, 1:]
        },
    ))
    
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# 4. 모델 관련 클래스 및 함수 정의
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# 5. 학습 관련 함수 정의
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

def train_model(dataset):
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset):
            train_step(inp['inputs'], tar['outputs'])

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

# 6. 메인 실행 함수
def main():
    # 데이터 준비
    train_data = load_data()
    questions, answers, tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE, MAX_LENGTH = prepare_data(train_data)
    dataset = create_dataset(questions, answers, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH)
    
    # 모델 및 옵티마이저 설정
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # 학습 실행
    train_model(dataset)
    
    # 모델 평가 및 테스트
    test_model()

def positional_encoding(position, d_model):
    # 위치별 각도 계산
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    # 짝수 인덱스에는 sin 적용
    sines = np.sin(angle_rads[:, 0::2])
    
    # 홀수 인덱스에는 cos 적용
    cosines = np.cos(angle_rads[:, 1::2])
    
    # sin과 cos 값을 교차로 쌓기
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    
    # 배치 차원 추가
    pos_encoding = pos_encoding[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

if __name__ == "__main__":
    main() 