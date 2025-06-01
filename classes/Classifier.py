import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import pickle
from sklearn.metrics import classification_report
import pandas as pd

class Classifier:
    def __init__(self, input_shape=(46,), num_classes=3, model_path='classifier_model.h5'):
        """
        Initialize the Improved Classifier with better architecture for similar classes.
        
        Args:
            input_shape: Shape of input vectors (default: (46,))
            num_classes: Number of classes to predict (default: 3)
            model_path: Path to save/load model weights
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()  # Add feature scaling

    def build_model(self):
        """Build an improved model architecture for similar fish classification"""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Feature extraction with residual connections
        x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # First residual block
        residual1 = x
        x = layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual1])  # Residual connection
        x = layers.Dropout(0.3)(x)
        
        # Second block with dimensionality reduction
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)
        
        # Third block - deeper feature learning
        residual2 = x
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual2])  # Second residual connection
        x = layers.Dropout(0.35)(x)
        
        # Final feature extraction
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Classification head
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)  
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Use Adam with cyclical learning rate
        initial_learning_rate = 0.001
        
        # Compatible optimizer setup
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Simple but effective compilation
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_data(self, h5_path):
        """
        Load data from HDF5 file and prepare for training with preprocessing.
        """
        with h5py.File(h5_path, 'r') as f:
            X = np.array(f['X'])
            y = np.array(f['y'])
            class_mapping = eval(f.attrs['class_mapping'])

        # Feature scaling - important for similar classes
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels to numerical values
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Stratified split with larger validation set for better evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_val, y_train, y_val, class_mapping
    
    def create_advanced_callbacks(self, patience=15):
        """Create advanced callbacks for better training"""
        callbacks = [
            # Early stopping with more patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=patience,
                restore_best_weights=True,
                min_delta=0.001,
                mode='max'
            ),
            
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  # Less aggressive reduction
                patience=8,
                min_lr=1e-7,
                verbose=1,
                cooldown=3
            ),
            
            # Cyclical learning rate for better convergence
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (0.95 ** (epoch // 10)),
                verbose=0
            )
        ]
        
        return callbacks
    
    def train(self, h5_path, epochs=100, batch_size=32, patience=15, save_model=True):
        """
        Train the classifier with improved strategy.
        """
        # Load and prepare data
        X_train, X_val, y_train, y_val, class_mapping = self.load_data(h5_path)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
            
        # Print model summary
        self.model.summary()
        
        # Create callbacks
        callbacks = self.create_advanced_callbacks(patience)
        
        # Calculate class weights for imbalanced data
        class_weights = self.calculate_class_weights(y_train)
        print(f"Class weights: {class_weights}")
        
        # Train the model with more epochs and class weights
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            shuffle=True
        )
        
        # Save the model and preprocessing objects
        if save_model:
            self.model.save(self.model_path)
            
            # Save label encoder and scaler
            encoder_path = Path(self.model_path).with_suffix('.encoder.pkl')
            scaler_path = Path(self.model_path).with_suffix('.scaler.pkl')
            
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            print(f"Model saved to {self.model_path}")
            print(f"Label encoder saved to {encoder_path}")
            print(f"Scaler saved to {scaler_path}")
        
        # Comprehensive evaluation
        self.advanced_evaluation(X_val, y_val)
        
        return history
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights to handle imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        return dict(enumerate(class_weights))
    
    def advanced_evaluation(self, X_val, y_val):
        """Comprehensive model evaluation"""
        print("\n=== Advanced Evaluation ===")

        # Get predictions
        y_pred_probs = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # Classification report with error handling
        try:
            # Ensure target names are strings
            target_names = [str(name) for name in self.label_encoder.classes_]
            if len(target_names) > 0:
                report = classification_report(
                    y_val,
                    y_pred_classes,
                    target_names=target_names,
                    output_dict=False
                )
                print("\nClassification Report:")
                print(report)
            else:
                print("\nClassification Report (no target names):")
                report = classification_report(y_val, y_pred_classes, output_dict=False)
                print(report)
        except Exception as e:
            print(f"\nError generating classification report: {e}")
            print("Basic accuracy:", np.mean(y_val == y_pred_classes))
        
        # Confusion matrix analysis
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Per-class confidence analysis
        self.analyze_confidence(y_val, y_pred_classes, y_pred_probs)
        
        # Misclassification analysis
        self.analyze_misclassifications(X_val, y_val, y_pred_classes, y_pred_probs)
    
    def analyze_confidence(self, y_true, y_pred, y_pred_probs):
        """Analyze prediction confidence patterns"""
        print(f"\n=== Confidence Analysis ===")
        
        confidence_data = []
        for i, (true_class, pred_class, probs) in enumerate(zip(y_true, y_pred, y_pred_probs)):
            confidence = probs[pred_class]
            is_correct = true_class == pred_class
            
            # Safe class name retrieval
            try:
                true_class_name = str(self.label_encoder.classes_[true_class])
                pred_class_name = str(self.label_encoder.classes_[pred_class])
            except (IndexError, AttributeError):
                true_class_name = f"Class_{true_class}"
                pred_class_name = f"Class_{pred_class}"
            
            confidence_data.append({
                'true_class': true_class_name,
                'predicted_class': pred_class_name,
                'confidence': confidence,
                'correct': is_correct,
                'max_prob': np.max(probs),
                'entropy': -np.sum(probs * np.log(probs + 1e-8))
            })
        
        confidence_df = pd.DataFrame(confidence_data)
        
        print(f"Overall accuracy: {confidence_df['correct'].mean():.3f}")
        print(f"Average confidence: {confidence_df['confidence'].mean():.3f}")
        print(f"Correct predictions confidence: {confidence_df[confidence_df['correct']]['confidence'].mean():.3f}")
        print(f"Wrong predictions confidence: {confidence_df[~confidence_df['correct']]['confidence'].mean():.3f}")
        print(f"Average entropy: {confidence_df['entropy'].mean():.3f}")
        
        # Low confidence predictions (potential hard cases)
        low_conf = confidence_df[confidence_df['confidence'] < 0.7]
        print(f"\nLow confidence predictions (< 0.7): {len(low_conf)} samples")
        if len(low_conf) > 0:
            print(f"Accuracy on low confidence: {low_conf['correct'].mean():.3f}")
    
    def analyze_misclassifications(self, X_val, y_true, y_pred, y_pred_probs):
        """Analyze misclassification patterns"""
        print(f"\n=== Misclassification Analysis ===")
        
        misclassified = y_true != y_pred
        if np.sum(misclassified) == 0:
            print("No misclassifications found!")
            return
            
        print(f"Total misclassified: {np.sum(misclassified)}")
        
        # Find most common misclassification pairs with error handling
        misclass_pairs = {}
        for true_idx, pred_idx in zip(y_true[misclassified], y_pred[misclassified]):
            try:
                true_class = str(self.label_encoder.classes_[true_idx])
                pred_class = str(self.label_encoder.classes_[pred_idx])
                pair = (true_class, pred_class)
                misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
            except (IndexError, AttributeError):
                # Fallback to index-based names if classes not available
                pair = (f"Class_{true_idx}", f"Class_{pred_idx}")
                misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
        
        print("\nMost common misclassification pairs:")
        for pair, count in sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pair[0]} -> {pair[1]}: {count} times")
    
    def load_model(self):
        """Load a pre-trained model and all preprocessing objects"""
        model_path = Path(self.model_path)
        encoder_path = model_path.with_suffix('.encoder.pkl')
        scaler_path = model_path.with_suffix('.scaler.pkl')
        
        if model_path.exists() and encoder_path.exists() and scaler_path.exists():
            self.model = models.load_model(self.model_path)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            return True
        return False
    
    def predict(self, mask_vector):
        """
        Predict the class of a fish mask vector with preprocessing.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        # Preprocess the input
        if len(mask_vector.shape) == 1:
            mask_vector = np.expand_dims(mask_vector, axis=0)
        
        # Apply the same scaling as during training
        mask_vector_scaled = self.scaler.transform(mask_vector)
        
        # Make prediction
        probabilities = self.model.predict(mask_vector_scaled, verbose=0)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_probability = float(probabilities[predicted_class_idx])
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Return top predictions for similar classes
        top_k = min(3, self.num_classes)
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = []
        for idx in top_indices:
            class_name = self.label_encoder.inverse_transform([idx])[0]
            probability = float(probabilities[idx])
            top_predictions.append((class_name, probability))
        
        return predicted_class, predicted_probability, top_predictions

    def evaluate(self, h5_path):
        """
        Evaluate the model on test data.
        
        Args:
            h5_path: Path to HDF5 test data
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        # Load data
        X_train, X_val, y_train, y_val, class_mapping = self.load_data(h5_path)
        
        # Evaluate on validation set
        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        return {'loss': loss, 'accuracy': accuracy}