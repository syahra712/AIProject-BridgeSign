import pickle
import os
import shutil
import numpy as np
import tensorflow as tf

# Define paths
INPUT_MODEL_PATH = '/Users/admin/Desktop/American-Sign-language-Detection-System/modelbest.p'
OUTPUT_MODEL_PATH = '/Users/admin/Desktop/American-Sign-language-Detection-System/modelbestestcnn.p'
LABEL_ENCODER_PATH = '/Users/admin/Desktop/American-Sign-language-Detection-System/label_encoder.pickle'

def convert_model():
    """Convert modelbest.p to model.p and verify compatibility."""
    try:
        # Check if input model file exists
        if not os.path.exists(INPUT_MODEL_PATH):
            print(f"❌ Error: Input model file {INPUT_MODEL_PATH} does not exist.")
            return

        # Load the model from modelbest.p
        with open(INPUT_MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)

        # Verify the model dictionary structure
        if 'model' not in model_dict:
            print(f"❌ Error: {INPUT_MODEL_PATH} does not contain a 'model' key.")
            return

        model = model_dict['model']
        print("✅ Model loaded successfully from:", INPUT_MODEL_PATH)

        # Verify the model type (TensorFlow/Keras for CNN)
        if not isinstance(model, tf.keras.models.Sequential):
            print(f"⚠️ Warning: Loaded model is of type {type(model)}. Expected Keras Sequential model.")
        
        # Save the model to model.p
        with open(OUTPUT_MODEL_PATH, 'wb') as f:
            pickle.dump({'model': model}, f)
        print("✅ Model saved successfully as:", OUTPUT_MODEL_PATH)

        # Check for label encoder and copy it if necessary
        if os.path.exists(LABEL_ENCODER_PATH):
            output_label_encoder_path = os.path.join(
                os.path.dirname(OUTPUT_MODEL_PATH), 'label_encoder.pickle'
            )
            if not os.path.exists(output_label_encoder_path):
                shutil.copy(LABEL_ENCODER_PATH, output_label_encoder_path)
                print("✅ Label encoder copied to:", output_label_encoder_path)
        else:
            print(f"⚠️ Warning: Label encoder file {LABEL_ENCODER_PATH} not found. Ensure it exists for proper inference.")

        # Optional: Test model compatibility with a dummy input
        dummy_input = np.zeros((1, 42, 1))  # Shape expected by CNN
        try:
            prediction = model.predict(dummy_input, verbose=0)
            print(f"✅ Model test successful. Output shape: {prediction.shape}")
        except Exception as e:
            print(f"❌ Error testing model with dummy input: {e}")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")

if __name__ == "__main__":
    convert_model()
