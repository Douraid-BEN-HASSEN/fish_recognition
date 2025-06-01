import matplotlib.pyplot as plt
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes import Classifier

CLASSIFIER_PATH = './models/face_sardine_classifier/face_sardine_classifier.h5'
DATASET_PATH = './datasets/sardine_classifier_dataset/face_sardine_classifier_dataset.h5'

INPUT_SHAPE = (46,)
CLASSES = 3

EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 30

DEBUG = True

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{os.path.dirname(CLASSIFIER_PATH)}/training_metrics.png')
    plt.show()

# Initialize the classifier
classifier = Classifier(
    input_shape=INPUT_SHAPE,
    num_classes=CLASSES,
    model_path=CLASSIFIER_PATH
)

# Train the model
history = classifier.train(
    DATASET_PATH, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
    save_model=True
)

if DEBUG:
    plot_training_history(history)