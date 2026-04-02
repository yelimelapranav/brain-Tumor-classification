# brain-Tumor-classification 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, ResNet50
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# ================================
# 📂 PATHS
# ================================
train_path = "/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri/Training"
test_path = "/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri/Testing"

IMG_SIZE = (224,224)
BATCH_SIZE = 32   # safer for fusion
EPOCHS = 50

# ================================
# 📊 DATA
# ================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ================================
# 🧠 INPUT
# ================================
input_layer = layers.Input(shape=(224,224,3))

# ================================
# 🔥 BACKBONES (NO shared input!)
# ================================
eff_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
den_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
res_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze
for model_base in [eff_base, den_base, res_base]:
    model_base.trainable = False

# Pass SAME input through each model
eff_feat = eff_base(input_layer)
den_feat = den_base(input_layer)
res_feat = res_base(input_layer)

# ================================
# 📌 FEATURE EXTRACTION
# ================================
eff_feat = layers.GlobalAveragePooling2D()(eff_feat)
den_feat = layers.GlobalAveragePooling2D()(den_feat)
res_feat = layers.GlobalAveragePooling2D()(res_feat)

# ================================
# 🔗 FUSION
# ================================
fusion = layers.Concatenate()([eff_feat, den_feat, res_feat])

# ================================
# 🧠 ATTENTION
# ================================
attention = layers.Dense(fusion.shape[-1], activation='softmax')(fusion)
attended = layers.Multiply()([fusion, attention])

# ================================
# 🔥 CLASSIFIER
# ================================
x = layers.BatchNormalization()(attended)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs=input_layer, outputs=output)

# ================================
# ⚙️ COMPILE
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# 🏋️ TRAIN
# ================================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# ================================
# 📊 EVALUATION
# ================================
preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(test_data.classes, y_pred))

# ================================
# 📈 PLOT
# ================================
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Training Curve")
plt.show()

# ================================
# 💾 SAVE
# ================================
model.save("fusion_attention_model_fixed.h5")

print("\n🚀 Model trained successfully!")
Found 2870 images belonging to 4 classes.
Found 394 images belonging to 4 classes.
Model: "functional_5"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_8       │ (None, 224, 224,  │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ efficientnetb0      │ (None, 7, 7,      │  4,049,571 │ input_layer_8[0]… │
│ (Functional)        │ 1280)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ densenet121         │ (None, 7, 7,      │  7,037,504 │ input_layer_8[0]… │
│ (Functional)        │ 1024)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resnet50            │ (None, 7, 7,      │ 23,587,712 │ input_layer_8[0]… │
│ (Functional)        │ 2048)             │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 1280)      │          0 │ efficientnetb0[0… │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 1024)      │          0 │ densenet121[0][0] │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 2048)      │          0 │ resnet50[0][0]    │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_2       │ (None, 4352)      │          0 │ global_average_p… │
│ (Concatenate)       │                   │            │ global_average_p… │
│                     │                   │            │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_9 (Dense)     │ (None, 4352)      │ 18,944,256 │ concatenate_2[0]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multiply_2          │ (None, 4352)      │          0 │ concatenate_2[0]… │
│ (Multiply)          │                   │            │ dense_9[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 4352)      │     17,408 │ multiply_2[0][0]  │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_10 (Dense)    │ (None, 256)       │  1,114,368 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_2 (Dropout) │ (None, 256)       │          0 │ dense_10[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_11 (Dense)    │ (None, 4)         │      1,028 │ dropout_2[0][0]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 54,751,847 (208.86 MB)
 Trainable params: 20,068,356 (76.55 MB)
 Non-trainable params: 34,683,491 (132.31 MB)

 
📊 Classification Report:

              precision    recall  f1-score   support

           0       0.92      0.24      0.38       100
           1       0.60      0.98      0.75       115
           2       0.82      0.94      0.88       105
           3       0.88      0.70      0.78        74

    accuracy                           0.73       394
   macro avg       0.81      0.72      0.70       394
weighted avg       0.79      0.73      0.69       394

Model trained successfully!
