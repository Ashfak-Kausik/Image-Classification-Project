import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Assume images are organized in folders: 'data/train/defective', 'data/train/non-defective'
train_dir = 'data/train'
val_dir = 'data/validation'

# Image Augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

# Step 2: Building the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Model Training
history = model.fit(train_generator, validation_data=val_generator, epochs=10, steps_per_epoch=100, validation_steps=50)

# Step 4: Model Evaluation
val_generator.reset()
predictions = model.predict(val_generator, steps=len(val_generator), verbose=1)
predicted_classes = [1 if x > 0.5 else 0 for x in predictions]

# Confusion Matrix and Classification Report
true_classes = val_generator.classes
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
print(classification_report(true_classes, predicted_classes))

# Plotting accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()