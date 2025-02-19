import tensorflow as tf
import tensorflow_addons as tfa
import math
import os

from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Input, MaxPooling2D, 
    concatenate, Dropout, BatchNormalization, 
    Activation, Add, Multiply, Concatenate
)
from tensorflow.keras.models import Model

##############################################################################
# (예시) 임의로 정의해둔 부분. 실제 사용시 각자 환경에 맞춰 정의하세요.
##############################################################################

# 임의 숫자
num_train_examples = 1000
batch_size = 8

# 임의의 train/valid/test dataset
# 실제 환경에 맞게 tf.data.Dataset을 구성하세요.
def dummy_dataset(num_samples=100):
    """ 간단한 (image, mask) 형태 더미 데이터셋 생성 """
    for _ in range(num_samples):
        # 256x256 RGB
        image = tf.random.uniform([256, 256, 3], 0, 1)
        # 256x256 binary mask
        mask = tf.random.uniform([256, 256, 1], 0, 1)
        yield image, mask

train_dataset = tf.data.Dataset.from_generator(
    lambda: dummy_dataset(40),
    output_types=(tf.float32, tf.float32),
    output_shapes=((256,256,3),(256,256,1))
).batch(batch_size)

valid_dataset = tf.data.Dataset.from_generator(
    lambda: dummy_dataset(10),
    output_types=(tf.float32, tf.float32),
    output_shapes=((256,256,3),(256,256,1))
).batch(batch_size)

test_dataset = tf.data.Dataset.from_generator(
    lambda: dummy_dataset(10),
    output_types=(tf.float32, tf.float32),
    output_shapes=((256,256,3),(256,256,1))
).batch(batch_size)

# 아래도 예시로 만든 placeholder 함수들
def bce_dice_loss(y_true, y_pred):
    """Placeholder: 실제로는 사용자가 구현한 BCE+Dice 혼합 손실"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - (2.0 * tf.reduce_sum(y_true*y_pred) + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
    return bce + dice

def mean_iou(y_true, y_pred):
    """Placeholder: 실제론 tf.keras.metrics.MeanIoU 등을 활용하거나 threshold 과정 필요"""
    return 0.5  # 임의

def dice_coeff(y_true, y_pred):
    """Placeholder: 실제로는 threshold 등을 고려하여 Dice 계산"""
    return 0.5  # 임의

def compare_pretrained_models(model1, model2, dataset):
    """Placeholder: 두 모델의 성능을 비교하는 함수라고 가정"""
    return {"comparison": "Model2 is better, etc."}


##############################################################################
# 1. 데이터 Augmentation 강화
##############################################################################
def enhanced_augmentation(image, mask):
    # 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # 회전
    if tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform([], -20, 20) * math.pi / 180
        image = tfa.image.rotate(image, angle)
        mask = tfa.image.rotate(mask, angle)
    
    # 밝기 조정
    image = tf.image.random_brightness(image, 0.2)
    
    # 대비 조정
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # 노이즈 추가
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    
    return image, mask


##############################################################################
# 2. 학습률 스케줄링 최적화
##############################################################################
def get_optimized_lr_schedule():
    initial_lr = 1e-3
    min_lr = 1e-6
    # 한 "cycle"에 몇 스텝을 쓸지, 예시로 5에포크 단위
    decay_steps = (num_train_examples // batch_size) * 5
    
    lr_schedule = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=decay_steps,
        t_mul=2.0,
        m_mul=0.9,
        alpha=min_lr / initial_lr
    )
    return lr_schedule


##############################################################################
# 3. 손실 함수 개선 (기존 bce_dice + focal 혼합)
##############################################################################
def combined_loss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    
    # focal loss
    focal_loss = -alpha * (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred + 1e-7)
    focal_loss = tf.reduce_mean(focal_loss)
    
    # 기존 BCE+Dice
    bce_dice = bce_dice_loss(y_true, y_pred)
    
    return bce_dice + 0.5 * focal_loss


##############################################################################
# 4. 모델 아키텍처 개선: Enhanced Pretrained UNet (Attention + Skip)
##############################################################################
def create_enhanced_pretrained_unet(input_shape=(256, 256, 3)):
    """
    - VGG16의 block1_conv2, block2_conv2, block3_conv3, block4_conv3, block5_conv3를 활용
    - bottleneck: block5_conv3
    - skip: block4_conv3, block3_conv3, block2_conv2, block1_conv2
    - attention block 적용
    """
    # base_model
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # 원하는 레이어 출력만 뽑기
    skip1 = base_model.get_layer('block1_conv2').output  # shape: (H/2, W/2, 64)
    skip2 = base_model.get_layer('block2_conv2').output  # shape: (H/4, W/4, 128)
    skip3 = base_model.get_layer('block3_conv3').output  # shape: (H/8, W/8, 256)
    skip4 = base_model.get_layer('block4_conv3').output  # shape: (H/16, W/16, 512)
    bottleneck = base_model.get_layer('block5_conv3').output  # shape: (H/32, W/32, 512)
    
    encoder = tf.keras.Model(inputs=base_model.input, 
                             outputs=[skip1, skip2, skip3, skip4, bottleneck],
                             name="VGG16_Encoder")
    
    # Attention 블록 정의
    def attention_block(x, g, inter_channel):
        """
        x: skip (H, W, C1)
        g: gating (H, W, C2)  -> 이미 업샘플된 상태
        두 feature의 channel수를 inter_channel로 맞춘 뒤, Add -> sigmoid -> x * rate
        """
        # 1x1 conv로 채널 정렬
        theta_x = Conv2D(inter_channel, kernel_size=1)(x)  # (H, W, inter_channel)
        phi_g   = Conv2D(inter_channel, kernel_size=1)(g)  # (H, W, inter_channel)
        
        f = Activation('relu')(Add()([theta_x, phi_g]))
        psi_f = Conv2D(1, kernel_size=1)(f)
        rate = Activation('sigmoid')(psi_f)
        
        # x와 rate(0~1)를 element-wise 곱
        attn_x = Multiply()([x, rate])
        return attn_x
    
    # Decoder
    # encoder 출력 받기
    s1, s2, s3, s4, x = encoder(base_model.input, training=False)
    # x: bottleneck (H/32, W/32, 512)
    
    # 첫 업샘 + attention + concat with skip4
    x = Conv2DTranspose(512, 2, strides=2, padding='same')(x)  # (H/16, W/16, 512)
    attn4 = attention_block(s4, x, inter_channel=512)           # (H/16, W/16, 512)
    x = Concatenate()([x, attn4])                               # (H/16, W/16, 1024)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 두 번째 업샘 + attention + concat with skip3
    x = Conv2DTranspose(256, 2, strides=2, padding='same')(x)  # (H/8, W/8, 256)
    attn3 = attention_block(s3, x, inter_channel=256)           # (H/8, W/8, 256)
    x = Concatenate()([x, attn3])                               # (H/8, W/8, 512)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 세 번째 업샘 + attention + concat with skip2
    x = Conv2DTranspose(128, 2, strides=2, padding='same')(x)  # (H/4, W/4, 128)
    attn2 = attention_block(s2, x, inter_channel=128)           # (H/4, W/4, 128)
    x = Concatenate()([x, attn2])                               # (H/4, W/4, 256)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 네 번째 업샘 + attention + concat with skip1
    x = Conv2DTranspose(64, 2, strides=2, padding='same')(x)   # (H/2, W/2, 64)
    attn1 = attention_block(s1, x, inter_channel=64)            # (H/2, W/2, 64)
    x = Concatenate()([x, attn1])                               # (H/2, W/2, 128)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 마지막 업샘 (원본 해상도)
    x = Conv2DTranspose(64, 2, strides=2, padding='same')(x)   # (H, W, 64)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # 최종 출력
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    
    # 최종 모델
    model = Model(inputs=base_model.input, outputs=outputs, name="Enhanced_VGG16_UNet")
    
    # Encoder 파라미터 freeze 예시 (원하는 경우)
    for layer in base_model.layers:
        layer.trainable = False
    
    return model


##############################################################################
# 5. 학습 전략 최적화: Progressive Learning + ModelCheckpoint 등
##############################################################################
def train_enhanced_model():
    # 모델 생성
    model = create_enhanced_pretrained_unet(input_shape=(256, 256, 3))
    
    # Progressive Learning 설정
    image_sizes = [(128, 128), (192, 192), (256, 256)]
    epochs_per_stage = [1, 1, 2]  # 예시로 각 해상도에서 몇 epoch
    
    total_history = []
    
    for size, epochs in zip(image_sizes, epochs_per_stage):
        print(f"\n[INFO] Training at resolution {size} for {epochs} epochs")
        
        # 데이터셋 리사이즈 & augmentation
        train_ds = train_dataset.map(
            lambda x, y: (tf.image.resize(x, size), tf.image.resize(y, size))
        ).map(enhanced_augmentation).prefetch(1)
        
        valid_ds = valid_dataset.map(
            lambda x, y: (tf.image.resize(x, size), tf.image.resize(y, size))
        ).prefetch(1)
        
        # 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(get_optimized_lr_schedule()),
            loss=combined_loss,
            metrics=[mean_iou, dice_coeff]
        )
        
        # 콜백
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_enhanced.h5',
                save_best_only=True,
                monitor='val_mean_iou',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_mean_iou',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        # 학습
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=valid_ds,
            callbacks=callbacks
        )
        total_history.append(history)
    
    return model, total_history


##############################################################################
# 6. Test Time Augmentation (TTA)
##############################################################################
def tta_predict(model, image, num_augments=3):
    """ 
    단일 이미지(배치1) TTA 예시
    - flip, 회전 등 간단한 예시
    """
    predictions = []
    # 원본 예측
    pred0 = model.predict(image)
    predictions.append(pred0)
    
    # 다양한 augmentation 적용 후 예측 (예시)
    for _ in range(num_augments - 1):
        # 좌우 뒤집기
        aug_image = tf.image.flip_left_right(image)
        pred = model.predict(aug_image)
        # 뒤집은 축 되돌림
        pred = tf.image.flip_left_right(pred)
        predictions.append(pred)
        
        # 간단한 random rotate
        angle = tf.random.uniform([], -10, 10) * math.pi / 180
        aug_image = tfa.image.rotate(image, angle)
        pred = model.predict(aug_image)
        # 회전 복원(정확히 복원하려면 inverse rotate 필요, 여기서는 단순화)
        predictions.append(pred)
    
    # 앙상블(평균)
    mean_pred = tf.reduce_mean(predictions, axis=0)
    return mean_pred


##############################################################################
# 7. 전체 실행 및 평가
##############################################################################
def run_enhanced_training():
    print("[INFO] Starting enhanced training process...")
    # 모델 학습
    enhanced_model, all_history = train_enhanced_model()
    
    # (예시) 원본 모델과 성능 비교 -> 실제론 원본 모델을 미리 만들어 놓아야 합니다
    # 여기서는 placeholder만 사용
    dummy_original_model = enhanced_model  # 실제론 다른 모델일 것
    comparison_results = compare_pretrained_models(dummy_original_model, enhanced_model, test_dataset)
    
    print("[INFO] Training complete.")
    print("[INFO] Comparison Results:", comparison_results)
    
    return enhanced_model, all_history, comparison_results


##############################################################################
# 실행
##############################################################################
if __name__ == "__main__":
    enhanced_model, history, results = run_enhanced_training()

    # 테스트셋 일부 샘플로 TTA 시연
    for images, masks in test_dataset.take(1):
        # 첫 배치
        tta_out = tta_predict(enhanced_model, images[0:1])  # 첫 이미지만
        print("TTA output shape:", tta_out.shape)
        break
