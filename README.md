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
