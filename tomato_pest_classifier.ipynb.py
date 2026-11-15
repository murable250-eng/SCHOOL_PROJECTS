"""
Tomato pest classifier: frozen MobileNetV2 features + exactly two Dense layers.
Metrics: Accuracy, macro F1, macro one-vs-rest AUC.

Dataset layout expected (standard folder-per-class):
    DATA_DIR/
        class_a/*.jpg
        class_b/*.jpg
        ...

Public dataset options (download, unzip, point DATA_DIR at the folder):
  - Kaggle: search "tomato pests" / "tomato leaf pest" image datasets
  - PlantVillage (Tomato_Spider_mites_Two_spotted_spider_mite class)
  - IP102 (large insect pest dataset; filter to tomato-relevant classes)
Install: pip install tensorflow scikit-learn numpy
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

DATA_DIR = "tomato_pests"   # <- change to your dataset folder
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 15
SEED = 42

# ---- Data: 70% train / 15% val / 15% test ----
full = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="int", shuffle=True, seed=SEED)
class_names = full.class_names
n_classes = len(class_names)

n_batches = tf.data.experimental.cardinality(full).numpy()
n_train = int(0.7 * n_batches)
n_val = int(0.15 * n_batches)
train_ds = full.take(n_train)
val_ds = full.skip(n_train).take(n_val)
test_ds = full.skip(n_train + n_val)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ---- Model: frozen backbone -> Dense(256, relu) -> Dense(n_classes, softmax) ----
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
])

backbone = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
backbone.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)                    # Dense layer 1
outputs = layers.Dense(n_classes, activation="softmax")(x)     # Dense layer 2
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ---- Evaluation on held-out test set ----
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = model.predict(test_ds)
y_pred = y_prob.argmax(axis=1)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
if n_classes == 2:
    auc = roc_auc_score(y_true, y_prob[:, 1])
else:
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                        labels=list(range(n_classes)))

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 (macro): {f1:.4f}")
print(f"AUC (OvR macro): {auc:.4f}\n")
print(classification_report(y_true, y_pred, target_names=class_names))

model.save("tomato_pest_model.keras")
