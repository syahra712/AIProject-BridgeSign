import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set the path to the data.pickle file
DATA_PATH = '/Users/admin/Desktop/American-Sign-language-Detection-System/data.pickle'

# Load the dataset
with open(DATA_PATH, 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])  # Shape: (samples, 42)
labels = np.asarray(data_dict['labels'])

# Encode labels (convert class names to integers)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Save label encoder for inference
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)

# Reshape data for Conv1D (samples, 42, 1)
data = data.reshape(-1, 42, 1)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded
)

# Define the CNN model
model = Sequential([
    # Convolutional layers to capture spatial patterns in hand landmarks
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(42, 1)),
    BatchNormalization(),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Flatten for dense layers
    Flatten(),
    
    # Dense layers for classification
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'The accuracy of the CNN model on the test data is: {test_accuracy * 100:.2f}%')

# Save the model in a format compatible with godknows.py
with open('modelbest.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model and label encoder saved successfully.")
import os
class ModelLoader:
    """Handles loading and fallback for the ASL recognition model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.label_encoder = None
        self.expected_features = 42
        
        # Try to load the model from the specified path
        if model_path and os.path.exists(model_path):
            try:
                model_dict = pickle.load(open(model_path, 'rb'))
                self.model = model_dict['model']
                # Load label encoder
                label_encoder_path = 'label_encoder.pickle'
                if os.path.exists(label_encoder_path):
                    self.label_encoder = pickle.load(open(label_encoder_path, 'rb'))
                print("✅ Model loaded successfully from:", model_path)
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.create_fallback_model()
        else:
            # Try to find the model in common locations
            possible_paths = [
                "modelbest.p",
                os.path.join(os.path.dirname(__file__), "modelbest.p"),
                os.path.join(os.path.expanduser("~"), "Desktop", "American-Sign-language-Detection-System", "modelbest.p")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        model_dict = pickle.load(open(path, 'rb'))
                        self.model = model_dict['model']
                        label_encoder_path = os.path.join(os.path.dirname(os.path.dirname(path)), "label_encoder.pickle")
                        if os.path.exists(label_encoder_path):
                            self.label_encoder = pickle.load(open(label_encoder_path, 'rb'))
                        print("✅ Model loaded successfully from:", path)
                        break
                    except:
                        pass
            
            if self.model is None:
                print("❌ Model not found, using fallback model")
                self.create_fallback_model()
    
    def create_fallback_model(self):
        """Create a simple fallback model for demonstration"""
        class MockModel:
            def predict(self, data):
                return [np.random.randint(0, 38)]
        
        self.model = MockModel()
        self.label_encoder = None
        print("⚠️ Using fallback model - predictions will be random")
    
    def predict(self, features):
        """Make a prediction using the model"""
        # Ensure correct feature length
        if len(features) < self.expected_features:
            features.extend([0] * (self.expected_features - len(features)))
        elif len(features) > self.expected_features:
            features = features[:self.expected_features]
        
        try:
            # Reshape features for CNN input
            features = np.asarray(features).reshape(1, 42, 1)
            prediction_probs = self.model.predict(features, verbose=0)
            prediction_idx = np.argmax(prediction_probs, axis=1)[0]
            if self.label_encoder:
                # Decode the prediction to original label
                return int(self.label_encoder.inverse_transform([prediction_idx])[0])
            return prediction_idx
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0