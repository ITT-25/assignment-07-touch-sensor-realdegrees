from dataclasses import dataclass
import threading
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from pynput.keyboard import Controller, Key

Point = Tuple[float, float]
@dataclass
class Prediction:
    char: str
    confidence: float
    rasterized_image: np.ndarray

    def __repr__(self) -> str:
        return f"Prediction(char={self.char}, confidence={self.confidence:.2f})"


class Recognizer:
    def __init__(self, model_path: str = None, *, confidence_threshold: float, detection_timer: float, model_type: str = "full") -> None:
        self.model_type = model_type
        self.model_path = model_path or f"text_input_{model_type}.keras"
        self.confidence_threshold = confidence_threshold
        self.points: List[Point] = []
        self._timer: Optional[threading.Timer] = None
        self.model = self._load_or_train_model()
        self.prediction: Prediction = Prediction(char='', confidence=0.0, rasterized_image=np.zeros((28, 28, 1), dtype=np.float32))
        self.keyboard = Controller()
        self.detection_timer = detection_timer

    def _label_to_char(self, label: int) -> str:
        """Convert model label to character based on model type."""
        if self.model_type == "lower":
            # For lowercase-only model: labels 0-25 map to 'a'-'z'
            if 0 <= label <= 25:
                return chr(ord('a') + label)
            else:
                return '?'
        else:  # full model
            # EMNIST byclass mapping:
            # 0-9: digits '0'-'9'
            # 10-35: uppercase 'A'-'Z'
            # 36-61: lowercase 'a'-'z'
            if 0 <= label <= 9:
                return str(label)
            elif 10 <= label <= 35:
                return chr(ord('A') + label - 10)
            elif 36 <= label <= 61:
                return chr(ord('a') + label - 36)
            else:
                return '?'

    def _char_to_label(self, char: str) -> int:
        """Convert character to model label based on model type."""
        if self.model_type == "lower":
            # For lowercase-only model: 'a'-'z' map to labels 0-25
            if char.islower():
                return ord(char) - ord('a')
            else:
                return -1
        else:  # full model
            # EMNIST byclass mapping
            if char.isdigit():
                return int(char)
            elif char.isupper():
                return ord(char) - ord('A') + 10
            elif char.islower():
                return ord(char) - ord('a') + 36
            else:
                return -1

    def _press_key(self, char: str) -> None:
        """Press the appropriate key using pynput, handling capitalization."""
        if char.isdigit():
            # For digits, just press the key
            self.keyboard.press(char)
            self.keyboard.release(char)
        elif char.isupper():
            # For uppercase letters, press shift + letter
            self.keyboard.press(Key.shift)
            self.keyboard.press(char.lower())
            self.keyboard.release(char.lower())
            self.keyboard.release(Key.shift)
        elif char.islower():
            # For lowercase letters, just press the key
            self.keyboard.press(char)
            self.keyboard.release(char)
        else:
            # Fallback for unknown characters
            self.keyboard.press(char)
            self.keyboard.release(char)

    def on_point(self, pt: Point) -> None:
        """Call this with each incoming (x,y) point."""
        self.points.append(pt)
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.detection_timer, self._on_inactivity)
        self._timer.start()

    def _on_inactivity(self) -> None:
        if not self.points:
            return
        img = self._preprocess_and_rasterize(self.points)
        # Apply the same transpose as in training data
        img = np.transpose(img)
        pred = self.model.predict(img[np.newaxis, ..., np.newaxis], verbose=0)[0]
        label = int(np.argmax(pred))
        confidence = float(pred[label])
        char = self._label_to_char(label)
        rasterized_image = img[np.newaxis, ..., np.newaxis]
        self.prediction = Prediction(char=char, confidence=confidence, rasterized_image=rasterized_image)
        print(f"Predicted: {self.prediction}")
        
        # Automatically press the key if confidence is high
        if confidence > self.confidence_threshold:
            self._press_key(char)
            print(f"Auto-pressed key: {char}")
        
        self.points.clear()

    def _preprocess_and_rasterize(self, pts: List[Point]) -> np.ndarray:
        """Normalize raw pts from pixel space to 28x28 image."""
        if not pts:
            return np.zeros((28, 28), dtype=np.float32)

        # Convert points to numpy array
        arr = np.array(pts, dtype=np.float32)
        
        # Normalize to fit in canvas
        min_xy = arr.min(axis=0)
        max_xy = arr.max(axis=0)
        arr -= min_xy
        
        # Scale to fit, maintaining aspect ratio
        size_xy = max_xy - min_xy
        scale = max(size_xy) if max(size_xy) > 0 else 1
        arr /= scale
        
        # Scale to fit in 24x24 area with centering
        arr *= 24
        
        # Center the letter in the 28x28 frame
        # Calculate the actual size after scaling
        scaled_min = arr.min(axis=0)
        scaled_max = arr.max(axis=0)
        scaled_size = scaled_max - scaled_min
        
        # Center horizontally and vertically
        center_offset = (28 - scaled_size) / 2
        arr += center_offset
        
        # Create image
        img = np.zeros((28, 28), dtype=np.float32)
        
        # Draw lines between consecutive points
        # Calculate threshold as 40% of the width
        threshold = 0.4 * 28
        
        for i in range(len(arr) - 1):
            x0, y0 = int(np.clip(arr[i, 0], 0, 27)), int(np.clip(arr[i, 1], 0, 27))
            x1, y1 = int(np.clip(arr[i+1, 0], 0, 27)), int(np.clip(arr[i+1, 1], 0, 27))
            
            # Only connect points if they're close enough
            distance = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            if distance <= threshold:
                cv2.line(img, (x0, y0), (x1, y1), 1.0, 2)
        
        # Apply Gaussian blur for smoother strokes
        img = cv2.GaussianBlur(img, (5, 5), 0.2)
        
        # match EMNIST orientation
        img = np.rot90(img) # Rotate 270 degrees (90 counter-clockwise)
        img = np.flipud(img) # Flip
        
        return np.clip(img, 0.0, 1.0)

    def _load_or_train_model(self) -> models.Model:
        try:
            model = models.load_model(self.model_path)
            print(f"Loaded existing {self.model_type} model from {self.model_path}")
            return model
        except Exception:
            print(f"Training new {self.model_type} EMNIST CNN model...")
            return self._train_model()

    def _train_model(self) -> models.Model:        
        # Load EMNIST data
        train, test = tfds.load(
            "emnist/byclass",
            split=["train", "test"],
            as_supervised=True
        )

        # Convert to numpy arrays
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        
        # Load a subset of data for faster training
        max_train_samples = 200000
        max_test_samples = 50000

        # Process training data based on model type
        for image, label in train:
            label_val = label.numpy()
            
            if self.model_type == "lower":
                # Only process lowercase letters (labels 36-61 in EMNIST byclass)
                if not (36 <= label_val <= 61):
                    continue
                # Remap to 0-25 for lowercase-only model
                label_val = label_val - 36
            # For full model, use all classes as-is
            
            # Process image
            img = tf.cast(image, tf.float32) / 255.0
            img = tf.image.transpose(img)
            img = img.numpy()
            
            # Add to training lists
            X_train_list.append(img)
            y_train_list.append(label_val)
            
            if len(X_train_list) >= max_train_samples:
                break

        # Process test data based on model type        
        for image, label in test:
            label_val = label.numpy()
            
            if self.model_type == "lower":
                # Only process lowercase letters (labels 36-61 in EMNIST byclass)
                if not (36 <= label_val <= 61):
                    continue
                # Remap to 0-25 for lowercase-only model
                label_val = label_val - 36
            # For full model, use all classes as-is
            
            img = tf.cast(image, tf.float32) / 255.0
            img = tf.image.transpose(img)
            img = img.numpy()
            
            X_test_list.append(img)
            y_test_list.append(label_val)
            
            if len(X_test_list) >= max_test_samples:
                break
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        
        # Set number of classes based on model type
        num_classes = 26 if self.model_type == "lower" else 62
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train_cat, test_size=0.2, random_state=42
        )
        
        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        print(f"Training {self.model_type} model with {num_classes} classes")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Define model architecture with appropriate output classes
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            # First conv block
            layers.Conv2D(32, 3, padding='same'),
            layers.LeakyReLU(alpha=0.1),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            
            # Second conv block
            layers.Conv2D(64, 3, padding='same'),
            layers.LeakyReLU(alpha=0.1),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128),
            layers.LeakyReLU(alpha=0.1),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        # Train the model with improved settings
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1
        )
        
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on test set
        y_test_cat = to_categorical(y_test, num_classes=num_classes)
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"{self.model_type.title()} model test accuracy: {test_acc*100:.2f}%")
        
        # Save model
        model.save(self.model_path)
        return model