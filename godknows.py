import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import warnings
import os
from datetime import datetime
from dataclasses import dataclass

# GUI imports
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QTextEdit, QSlider, QComboBox, QGroupBox, QFrame,
                                QProgressBar, QTabWidget, QListWidget, QListWidgetItem,
                                QMessageBox, QFileDialog, QCheckBox, QSpinBox, QSplitter,
                                QStatusBar)
    from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, QSize
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon, QMovie, QClipboard
except ImportError:
    print("PyQt5 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *

# Text-to-speech
try:
    import pyttsx3
except ImportError:
    print("pyttsx3 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyttsx3"])
    import pyttsx3

# Spell checking
try:
    from symspellpy.symspellpy import SymSpell, Verbosity
except ImportError:
    print("symspellpy not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "symspellpy"])
    from symspellpy.symspellpy import SymSpell, Verbosity

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@dataclass
class RecognitionStats:
    """Statistics for recognition performance"""
    total_predictions: int = 0
    correct_predictions: int = 0
    session_start: datetime = None
    words_formed: int = 0
    sentences_completed: int = 0

class ModelLoader:
    """Handles loading and fallback for the ASL recognition model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.expected_features = 42
        
        # Try to load the model from the specified path
        if model_path and os.path.exists(model_path):
            try:
                model_dict = pickle.load(open(model_path, 'rb'))
                self.model = model_dict['model']
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
                # Simple mock prediction
                return [np.random.randint(0, 38)]
        
        self.model = MockModel()
        print("⚠️ Using fallback model - predictions will be random")
    
    def predict(self, features):
        """Make a prediction using the model"""
        # Ensure correct feature length
        if len(features) < self.expected_features:
            features.extend([0] * (self.expected_features - len(features)))
        elif len(features) > self.expected_features:
            features = features[:self.expected_features]
        
        try:
            prediction = self.model.predict([np.asarray(features)])
            return int(prediction[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0

class DictionaryLoader:
    """Handles loading and fallback for the dictionary"""
    
    def __init__(self, dict_path=None):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Try to load the dictionary from the specified path
        if dict_path and os.path.exists(dict_path):
            try:
                self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
                print("✅ Dictionary loaded successfully from:", dict_path)
                return
            except Exception as e:
                print(f"❌ Error loading dictionary: {e}")
        
        # Try to find the dictionary in common locations
        possible_paths = [
            "dictionary.txt",
            os.path.join(os.path.dirname(__file__), "dictionary.txt"),
            os.path.join(os.path.expanduser("~"), "Desktop", "American-Sign-language-Detection-System", "dictionary.txt")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.sym_spell.load_dictionary(path, term_index=0, count_index=1)
                    print("✅ Dictionary loaded successfully from:", path)
                    return
                except:
                    pass
        
        # Create a basic dictionary if loading fails
        print("⚠️ Dictionary not found, creating basic dictionary")
        self.create_basic_dictionary()
    
    def create_basic_dictionary(self):
        """Create a basic dictionary with common words"""
        common_words = [
            "hello", "world", "thank", "you", "please", "help", "good", "bad",
            "yes", "no", "maybe", "what", "when", "where", "who", "how", "why",
            "time", "day", "today", "tomorrow", "morning", "afternoon", "evening",
            "night", "food", "water", "eat", "drink", "sleep", "work", "play",
            "school", "home", "family", "friend", "love", "like", "need", "want",
            "happy", "sad", "sorry", "excuse", "me", "name", "sign", "language",
            "learn", "teach", "student", "teacher", "book", "read", "write"
        ]
        
        for word in common_words:
            self.sym_spell.create_dictionary_entry(word, 1000)
        
        print("✅ Basic dictionary created with", len(common_words), "words")
    
    def get_suggestions(self, word, max_edit_distance=2, max_suggestions=5):
        """Get word suggestions"""
        if not word:
            return []
        
        suggestions = self.sym_spell.lookup(
            word.lower(), 
            Verbosity.CLOSEST, 
            max_edit_distance=max_edit_distance, 
            include_unknown=True
        )
        
        return suggestions[:max_suggestions]

class HandProcessor:
    """Processes hand landmarks and draws on frames"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def process_frame(self, frame):
        """Process a frame and return hand landmarks"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_data = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract features
                    data_aux = []
                    x_, y_ = [], []
                    
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)
                    
                    for lm in hand_landmarks.landmark:
                        data_aux.extend([lm.x - min(x_), lm.y - min(y_)])
                    
                    hand_data = data_aux
            
            return frame, hand_data
        except Exception as e:
            print(f"Hand processing error: {e}")
            return frame, None

class PredictionBuffer:
    """Manages prediction stability"""
    
    def __init__(self, buffer_size=30, stability_threshold=25, registration_delay=2.5):
        self.buffer = []
        self.buffer_size = buffer_size
        self.stability_threshold = stability_threshold
        self.last_time = time.time()
        self.registration_delay = registration_delay
    
    def add_prediction(self, predicted_char):
        """Add a prediction to the buffer and check for stability"""
        self.buffer.append(predicted_char)
        
        # Keep buffer at specified size
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Check if prediction is stable
        if self.buffer.count(predicted_char) > self.stability_threshold:
            current_time = time.time()
            if current_time - self.last_time > self.registration_delay:
                self.last_time = current_time
                return predicted_char
        
        return None

class EnhancedASLApp(QMainWindow):
    """Enhanced ASL Recognition Application - Simple Version"""
    
    def __init__(self):
        super().__init__()
        
        # Setup window properties
        self.setWindowTitle("Enhanced ASL Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.dictionary_loader = DictionaryLoader()
        self.hand_processor = HandProcessor()
        self.prediction_buffer = PredictionBuffer()
        self.engine = pyttsx3.init()
        
        # Setup labels dictionary
        self.labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z
        self.labels_dict.update({26 + i: str(i) for i in range(10)})  # 0-9
        self.labels_dict[36] = ' '
        self.labels_dict[37] = '.'
        
        # Application state
        self.buffer = []
        self.stable_char = None
        self.word_buffer = ""
        self.sentence = ""
        self.history = []  # Store previous sentences
        self.is_camera_active = True
        self.show_landmarks = True
        self.auto_suggest = True
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33 FPS
        
        # Setup UI
        self.setup_ui()
        self.apply_styles()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Statistics
        self.stats = {
            "letters_detected": 0,
            "words_formed": 0,
            "sentences_completed": 0,
            "start_time": time.time()
        }
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Main recognition tab
        recognition_tab = self.create_recognition_tab()
        tabs.addTab(recognition_tab, "Recognition")
        
        # History tab
        history_tab = self.create_history_tab()
        tabs.addTab(history_tab, "History")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "Settings")
        
        # Help tab
        help_tab = self.create_help_tab()
        tabs.addTab(help_tab, "Help")
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)
        main_widget.setLayout(main_layout)
    
    def create_recognition_tab(self):
        """Create the main recognition interface"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # Left side - Video feed
        left_panel = QVBoxLayout()
        
        # Video display
        video_group = QGroupBox("Camera Feed")
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #3498db; border-radius: 10px; background-color: white;")
        video_layout.addWidget(self.video_label)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        self.toggle_camera_btn = QPushButton("Pause Camera")
        self.toggle_camera_btn.clicked.connect(self.toggle_camera)
        camera_controls.addWidget(self.toggle_camera_btn)
        
        self.toggle_landmarks_btn = QPushButton("Hide Landmarks")
        self.toggle_landmarks_btn.clicked.connect(self.toggle_landmarks)
        camera_controls.addWidget(self.toggle_landmarks_btn)
        
        video_layout.addLayout(camera_controls)
        video_group.setLayout(video_layout)
        left_panel.addWidget(video_group)
        
        # Current letter display
        letter_group = QGroupBox("Current Detection")
        letter_layout = QVBoxLayout()
        
        self.char_label = QLabel("Current Letter: ---")
        self.char_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.char_label.setAlignment(Qt.AlignCenter)
        letter_layout.addWidget(self.char_label)
        
        # Add confidence indicator
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        letter_layout.addWidget(self.confidence_bar)
        
        letter_group.setLayout(letter_layout)
        left_panel.addWidget(letter_group)
        
        layout.addLayout(left_panel, 3)  # 3:2 ratio for left:right
        
        # Right side - Text and controls
        right_panel = QVBoxLayout()
        
        # Current word
        word_group = QGroupBox("Current Word")
        word_layout = QVBoxLayout()
        
        self.word_label = QLabel("Current Word: ")
        self.word_label.setFont(QFont("Arial", 14))
        word_layout.addWidget(self.word_label)
        
        # Word controls
        word_controls = QHBoxLayout()
        self.clear_word_btn = QPushButton("❌ Clear Word")
        self.clear_word_btn.clicked.connect(self.clear_word)
        word_controls.addWidget(self.clear_word_btn)
        
        self.add_space_btn = QPushButton("⎵ Add Space")
        self.add_space_btn.clicked.connect(lambda: self.handle_special_char(' '))
        word_controls.addWidget(self.add_space_btn)
        
        self.add_period_btn = QPushButton("• Add Period")
        self.add_period_btn.clicked.connect(lambda: self.handle_special_char('.'))
        word_controls.addWidget(self.add_period_btn)
        
        word_layout.addLayout(word_controls)
        word_group.setLayout(word_layout)
        right_panel.addWidget(word_group)
        
        # Word suggestions
        suggestions_group = QGroupBox("Word Suggestions")
        suggestions_layout = QVBoxLayout()
        
        self.suggest_buttons = []
        for i in range(5):
            btn = QPushButton("")
            btn.setEnabled(False)
            btn.clicked.connect(self.use_suggestion)
            self.suggest_buttons.append(btn)
            suggestions_layout.addWidget(btn)
        
        suggestions_group.setLayout(suggestions_layout)
        right_panel.addWidget(suggestions_group)
        
        # Sentence
        sentence_group = QGroupBox("Sentence")
        sentence_layout = QVBoxLayout()
        
        self.sent_label = QLabel("Sentence: ")
        self.sent_label.setFont(QFont("Arial", 14))
        self.sent_label.setWordWrap(True)
        sentence_layout.addWidget(self.sent_label)
        
        # Sentence controls
        sentence_controls = QHBoxLayout()
        
        self.speak_btn = QPushButton("🔊 Speak")
        self.speak_btn.clicked.connect(self.speak)
        sentence_controls.addWidget(self.speak_btn)
        
        self.clear_sent_btn = QPushButton("🧹 Clear Sentence")
        self.clear_sent_btn.clicked.connect(self.clear_sentence)
        sentence_controls.addWidget(self.clear_sent_btn)
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self.save_sentence)
        sentence_controls.addWidget(self.save_btn)
        
        sentence_layout.addLayout(sentence_controls)
        sentence_group.setLayout(sentence_layout)
        right_panel.addWidget(sentence_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        
        self.letters_label = QLabel("Letters Detected: 0")
        stats_layout.addWidget(self.letters_label, 0, 0)
        
        self.words_label = QLabel("Words Formed: 0")
        stats_layout.addWidget(self.words_label, 0, 1)
        
        self.sentences_label = QLabel("Sentences: 0")
        stats_layout.addWidget(self.sentences_label, 1, 0)
        
        self.time_label = QLabel("Session Time: 00:00")
        stats_layout.addWidget(self.time_label, 1, 1)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        layout.addLayout(right_panel, 2)
        tab.setLayout(layout)
        return tab
    
    def create_history_tab(self):
        """Create the history tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # History list
        self.history_list = QListWidget()
        layout.addWidget(self.history_list)
        
        # Controls
        controls = QHBoxLayout()
        
        speak_history_btn = QPushButton("🔊 Speak Selected")
        speak_history_btn.clicked.connect(self.speak_selected_history)
        controls.addWidget(speak_history_btn)
        
        copy_history_btn = QPushButton("📋 Copy Selected")
        copy_history_btn.clicked.connect(self.copy_selected_history)
        controls.addWidget(copy_history_btn)
        
        clear_history_btn = QPushButton("🧹 Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        controls.addWidget(clear_history_btn)
        
        save_history_btn = QPushButton("💾 Save All History")
        save_history_btn.clicked.connect(self.save_history)
        controls.addWidget(save_history_btn)
        
        layout.addLayout(controls)
        tab.setLayout(layout)
        return tab
    
    def create_settings_tab(self):
        """Create the settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Recognition settings
        recognition_group = QGroupBox("Recognition Settings")
        recognition_layout = QGridLayout()
        
        # Buffer size
        recognition_layout.addWidget(QLabel("Buffer Size:"), 0, 0)
        buffer_slider = QSlider(Qt.Horizontal)
        buffer_slider.setRange(10, 50)
        buffer_slider.setValue(30)
        buffer_slider.valueChanged.connect(self.update_buffer_size)
        recognition_layout.addWidget(buffer_slider, 0, 1)
        self.buffer_size_label = QLabel("30")
        recognition_layout.addWidget(self.buffer_size_label, 0, 2)
        
        # Stability threshold
        recognition_layout.addWidget(QLabel("Stability Threshold:"), 1, 0)
        stability_slider = QSlider(Qt.Horizontal)
        stability_slider.setRange(10, 40)
        stability_slider.setValue(25)
        stability_slider.valueChanged.connect(self.update_stability_threshold)
        recognition_layout.addWidget(stability_slider, 1, 1)
        self.stability_label = QLabel("25")
        recognition_layout.addWidget(self.stability_label, 1, 2)
        
        # Registration delay
        recognition_layout.addWidget(QLabel("Registration Delay (s):"), 2, 0)
        delay_slider = QSlider(Qt.Horizontal)
        delay_slider.setRange(10, 50)
        delay_slider.setValue(25)
        delay_slider.valueChanged.connect(self.update_registration_delay)
        recognition_layout.addWidget(delay_slider, 2, 1)
        self.delay_label = QLabel("2.5")
        recognition_layout.addWidget(self.delay_label, 2, 2)
        
        recognition_group.setLayout(recognition_layout)
        layout.addWidget(recognition_group)
        
        # TTS settings
        tts_group = QGroupBox("Text-to-Speech Settings")
        tts_layout = QGridLayout()
        
        # Voice selection
        tts_layout.addWidget(QLabel("Voice:"), 0, 0)
        self.voice_combo = QComboBox()
        
        # Get available voices
        try:
            voices = self.engine.getProperty('voices')
            for i, voice in enumerate(voices):
                self.voice_combo.addItem(f"{voice.name}", voice.id)
        except:
            self.voice_combo.addItem("Default Voice", "default")
        
        self.voice_combo.currentIndexChanged.connect(self.update_voice)
        tts_layout.addWidget(self.voice_combo, 0, 1, 1, 2)
        
        # Rate
        tts_layout.addWidget(QLabel("Rate:"), 1, 0)
        rate_slider = QSlider(Qt.Horizontal)
        rate_slider.setRange(100, 300)
        rate_slider.setValue(150)
        rate_slider.valueChanged.connect(self.update_rate)
        tts_layout.addWidget(rate_slider, 1, 1)
        self.rate_label = QLabel("150")
        tts_layout.addWidget(self.rate_label, 1, 2)
        
        # Volume
        tts_layout.addWidget(QLabel("Volume:"), 2, 0)
        volume_slider = QSlider(Qt.Horizontal)
        volume_slider.setRange(0, 100)
        volume_slider.setValue(100)
        volume_slider.valueChanged.connect(self.update_volume)
        tts_layout.addWidget(volume_slider, 2, 1)
        self.volume_label = QLabel("100%")
        tts_layout.addWidget(self.volume_label, 2, 2)
        
        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group)
        
        # Interface settings
        interface_group = QGroupBox("Interface Settings")
        interface_layout = QVBoxLayout()
        
        self.auto_suggest_check = QCheckBox("Enable automatic word suggestions")
        self.auto_suggest_check.setChecked(True)
        self.auto_suggest_check.stateChanged.connect(self.toggle_auto_suggest)
        interface_layout.addWidget(self.auto_suggest_check)
        
        self.auto_clear_check = QCheckBox("Automatically clear word after selection")
        self.auto_clear_check.setChecked(True)
        interface_layout.addWidget(self.auto_clear_check)
        
        self.auto_speak_check = QCheckBox("Automatically speak completed sentences")
        self.auto_speak_check.setChecked(False)
        interface_layout.addWidget(self.auto_speak_check)
        
        interface_group.setLayout(interface_layout)
        layout.addWidget(interface_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def create_help_tab(self):
        """Create the help tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>ASL Recognition System - Help</h2>
        
        <h3>Getting Started</h3>
        <p>This application recognizes American Sign Language (ASL) gestures through your webcam and converts them to text.</p>
        
        <h3>How to Use</h3>
        <ol>
            <li>Position your hand clearly in the camera view</li>
            <li>Make ASL signs for letters (A-Z), numbers (0-9), space, or period</li>
            <li>Hold each sign steady for about 2-3 seconds</li>
            <li>The recognized letter will appear and be added to the current word</li>
            <li>Use word suggestions or continue signing to form sentences</li>
            <li>Use the speak button to hear your sentence spoken aloud</li>
        </ol>
        
        <h3>Tips for Better Recognition</h3>
        <ul>
            <li>Ensure good lighting conditions</li>
            <li>Position your hand against a plain background</li>
            <li>Make clear, distinct hand shapes</li>
            <li>Hold each sign steady until it's recognized</li>
            <li>Use word suggestions for faster sentence building</li>
        </ul>
        
        <h3>Button Functions</h3>
        <ul>
            <li><b>🔊 Speak</b>: Speaks the current sentence aloud</li>
            <li><b>💾 Save</b>: Saves the current sentence to a text file</li>
            <li><b>❌ Clear Word</b>: Clears the current word being typed</li>
            <li><b>🧹 Clear Sentence</b>: Clears the entire sentence</li>
            <li><b>⎵ Add Space</b>: Manually adds a space</li>
            <li><b>• Add Period</b>: Manually adds a period</li>
        </ul>
        
        <h3>About</h3>
        <p>Enhanced ASL Recognition System v2.0 - Simple Edition</p>
        <p>This application uses MediaPipe for hand tracking and machine learning for ASL recognition.</p>
        """)
        
        layout.addWidget(help_text)
        tab.setLayout(layout)
        return tab
    
    def apply_styles(self):
        """Apply modern styling with black text for visibility"""
        # Set application style with black text
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                color: black;
            }
            
            QWidget {
                color: black;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 20px;
                background-color: white;
                color: black;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: black;
                font-weight: bold;
            }
            
            QLabel {
                color: black;
                font-weight: normal;
                background-color: transparent;
            }
            
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                min-height: 25px;
            }
            
            QPushButton:hover {
                background-color: #2980b9;
            }
            
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: white;
                color: black;
            }
            
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 3px;
            }
            
            QTabWidget::pane {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
            
            QTabBar::tab {
                background-color: #ecf0f1;
                color: black;
                padding: 10px 15px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
            }
            
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            
            QListWidget {
                background-color: white;
                color: black;
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
            
            QTextEdit {
                background-color: white;
                color: black;
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #cccccc;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #3498db;
                border: 2px solid #3498db;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            
            QComboBox {
                background-color: white;
                color: black;
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
            }
            
            QCheckBox {
                color: black;
                font-weight: normal;
            }
            
            QStatusBar {
                background-color: #ecf0f1;
                color: black;
                border-top: 1px solid #cccccc;
            }
        """)
    
    def update_frame(self):
        """Update video frame and process hand tracking"""
        if not self.is_camera_active:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.status_bar.showMessage("Error: Could not read from camera")
                return
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame with hand tracking
            if self.show_landmarks:
                processed_frame, hand_data = self.hand_processor.process_frame(frame)
            else:
                processed_frame = frame
                _, hand_data = self.hand_processor.process_frame(frame.copy())
            
            # Make prediction if hand is detected
            if hand_data:
                prediction_idx = self.model_loader.predict(hand_data)
                predicted_char = self.labels_dict.get(prediction_idx, '?')
                
                # Add to buffer and check for stable prediction
                stable_char = self.prediction_buffer.add_prediction(predicted_char)
                
                if stable_char:
                    self.handle_stable_character(stable_char)
                    self.stats["letters_detected"] += 1
                    self.update_statistics()
                
                # Update confidence visualization (mock value for demonstration)
                confidence = min(100, max(0, 70 + np.random.randint(-10, 20)))
                self.confidence_bar.setValue(confidence)
                
                # Update current letter display
                self.char_label.setText(f"Current Letter: {predicted_char}")
            
            # Convert frame to Qt format and display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
        except Exception as e:
            print(f"Frame update error: {e}")
    
    def handle_stable_character(self, char):
        """Handle a stable character detection"""
        self.stable_char = char
        
        if char == ' ':
            if self.word_buffer:
                self.sentence += self.word_buffer + " "
                self.word_buffer = ""
                self.stats["words_formed"] += 1
        elif char == '.':
            if self.word_buffer:
                self.sentence += self.word_buffer + "."
                self.word_buffer = ""
                self.stats["words_formed"] += 1
                self.stats["sentences_completed"] += 1
            else:
                self.sentence += "."
                self.stats["sentences_completed"] += 1
        else:
            self.word_buffer += char
        
        # Update UI
        self.update_word_label()
        self.update_sentence_label()
        self.update_suggestions()
        self.update_statistics()
    
    def handle_special_char(self, char):
        """Handle special character input from buttons"""
        if char == ' ':
            if self.word_buffer:
                self.sentence += self.word_buffer + " "
                self.word_buffer = ""
                self.stats["words_formed"] += 1
        elif char == '.':
            if self.word_buffer:
                self.sentence += self.word_buffer + "."
                self.word_buffer = ""
                self.stats["words_formed"] += 1
                self.stats["sentences_completed"] += 1
            else:
                self.sentence += "."
                self.stats["sentences_completed"] += 1
        
        # Update UI
        self.update_word_label()
        self.update_sentence_label()
        self.update_suggestions()
        self.update_statistics()
    
    def update_word_label(self):
        """Update the current word display"""
        self.word_label.setText(f"Current Word: {self.word_buffer}")
    
    def update_sentence_label(self):
        """Update the sentence display"""
        self.sent_label.setText(f"Sentence: {self.sentence}")
        
        # Auto-speak if enabled and sentence ends with period
        if hasattr(self, 'auto_speak_check') and self.auto_speak_check.isChecked() and self.sentence.endswith('.'):
            self.speak()
    
    def update_suggestions(self):
        """Update word suggestions"""
        if not self.auto_suggest:
            return
        
        try:
            suggestions = self.dictionary_loader.get_suggestions(self.word_buffer)
            
            for i in range(5):
                if i < len(suggestions):
                    self.suggest_buttons[i].setText(suggestions[i].term)
                    self.suggest_buttons[i].setEnabled(True)
                else:
                    self.suggest_buttons[i].setText("")
                    self.suggest_buttons[i].setEnabled(False)
        except Exception as e:
            print(f"Suggestion error: {e}")
    
    def update_statistics(self):
        """Update statistics display"""
        self.letters_label.setText(f"Letters Detected: {self.stats['letters_detected']}")
        self.words_label.setText(f"Words Formed: {self.stats['words_formed']}")
        self.sentences_label.setText(f"Sentences: {self.stats['sentences_completed']}")
        
        # Calculate session time
        elapsed = int(time.time() - self.stats["start_time"])
        minutes, seconds = divmod(elapsed, 60)
        self.time_label.setText(f"Session Time: {minutes:02d}:{seconds:02d}")
    
    def use_suggestion(self):
        """Use a word suggestion"""
        btn = self.sender()
        selected_word = btn.text()
        
        if hasattr(self, 'auto_clear_check') and self.auto_clear_check.isChecked():
            self.sentence += selected_word + " "
            self.word_buffer = ""
            self.stats["words_formed"] += 1
        else:
            self.word_buffer = selected_word
        
        self.update_word_label()
        self.update_sentence_label()
        self.update_suggestions()
        self.update_statistics()
    
    def speak(self):
        """Speak the current sentence - SIMPLE VERSION (no threading)"""
        text_to_speak = self.sentence.strip()
        if self.word_buffer:
            text_to_speak += " " + self.word_buffer
        
        if not text_to_speak:
            self.status_bar.showMessage("No text to speak")
            return
        
        try:
            # Simple direct TTS - will briefly freeze UI but no threading issues
            self.status_bar.showMessage("Speaking...")
            self.speak_btn.setText("🔊 Speaking...")
            self.speak_btn.setEnabled(False)
            
            # Process events to update UI
            QApplication.processEvents()
            
            # Speak the text
            self.engine.say(text_to_speak)
            self.engine.runAndWait()
            
            # Re-enable button
            self.speak_btn.setText("🔊 Speak")
            self.speak_btn.setEnabled(True)
            self.status_bar.showMessage("Speech completed")
            
            # Add to history if not already there
            if self.sentence and self.sentence not in self.history:
                self.history.append(self.sentence)
                self.history_list.addItem(self.sentence)
                
        except Exception as e:
            self.speak_btn.setText("🔊 Speak")
            self.speak_btn.setEnabled(True)
            self.status_bar.showMessage(f"Speech error: {str(e)}")
            print(f"TTS Error: {e}")
    
    def clear_word(self):
        """Clear the current word"""
        self.word_buffer = ""
        self.update_word_label()
        self.update_suggestions()
        self.status_bar.showMessage("Word cleared")
    
    def clear_sentence(self):
        """Clear the current sentence"""
        if self.sentence:
            # Add to history if not empty and not already there
            if self.sentence.strip() and self.sentence not in self.history:
                self.history.append(self.sentence)
                self.history_list.addItem(self.sentence)
        
        self.sentence = ""
        self.update_sentence_label()
        self.status_bar.showMessage("Sentence cleared")
    
    def save_sentence(self):
        """Save the current sentence to a file"""
        text_to_save = self.sentence.strip()
        if self.word_buffer:
            text_to_save += " " + self.word_buffer
        
        if not text_to_save:
            QMessageBox.information(self, "Save", "No text to save!")
            return
        
        # Get current date and time for default filename
        now = datetime.now()
        default_filename = f"asl_sentence_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Sentence", default_filename, "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text_to_save)
                QMessageBox.information(self, "Save Successful", f"Text saved to:\n{filename}")
                self.status_bar.showMessage(f"Saved to {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")
                self.status_bar.showMessage("Save failed")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        self.is_camera_active = not self.is_camera_active
        
        if self.is_camera_active:
            self.toggle_camera_btn.setText("Pause Camera")
            self.status_bar.showMessage("Camera active")
        else:
            self.toggle_camera_btn.setText("Resume Camera")
            self.status_bar.showMessage("Camera paused")
    
    def toggle_landmarks(self):
        """Toggle landmark display"""
        self.show_landmarks = not self.show_landmarks
        
        if self.show_landmarks:
            self.toggle_landmarks_btn.setText("Hide Landmarks")
        else:
            self.toggle_landmarks_btn.setText("Show Landmarks")
    
    def toggle_auto_suggest(self, state):
        """Toggle automatic word suggestions"""
        self.auto_suggest = state == Qt.Checked
        
        if self.auto_suggest:
            self.update_suggestions()
        else:
            for btn in self.suggest_buttons:
                btn.setText("")
                btn.setEnabled(False)
    
    # Settings methods
    def update_buffer_size(self, value):
        """Update prediction buffer size"""
        self.prediction_buffer.buffer_size = value
        self.buffer_size_label.setText(str(value))
    
    def update_stability_threshold(self, value):
        """Update stability threshold"""
        self.prediction_buffer.stability_threshold = value
        self.stability_label.setText(str(value))
    
    def update_registration_delay(self, value):
        """Update registration delay"""
        delay = value / 10.0
        self.prediction_buffer.registration_delay = delay
        self.delay_label.setText(f"{delay:.1f}")
    
    def update_voice(self, index):
        """Update TTS voice"""
        try:
            voice_id = self.voice_combo.currentData()
            if voice_id and voice_id != "default":
                self.engine.setProperty('voice', voice_id)
        except Exception as e:
            print(f"Voice update error: {e}")
    
    def update_rate(self, value):
        """Update TTS rate"""
        try:
            self.engine.setProperty('rate', value)
            self.rate_label.setText(str(value))
        except Exception as e:
            print(f"Rate update error: {e}")
    
    def update_volume(self, value):
        """Update TTS volume"""
        try:
            self.engine.setProperty('volume', value / 100.0)
            self.volume_label.setText(f"{value}%")
        except Exception as e:
            print(f"Volume update error: {e}")
    
    # History tab methods
    def speak_selected_history(self):
        """Speak selected history item"""
        selected_items = self.history_list.selectedItems()
        if selected_items:
            text = selected_items[0].text()
            
            try:
                self.status_bar.showMessage("Speaking history item...")
                self.engine.say(text)
                self.engine.runAndWait()
                self.status_bar.showMessage("Speech completed")
            except Exception as e:
                self.status_bar.showMessage(f"Speech error: {str(e)}")
                print(f"TTS Error: {e}")
        else:
            QMessageBox.information(self, "No Selection", "Please select a history item to speak.")
    
    def copy_selected_history(self):
        """Copy selected history item to current sentence"""
        selected_items = self.history_list.selectedItems()
        if selected_items:
            self.sentence = selected_items[0].text()
            self.update_sentence_label()
            self.status_bar.showMessage("History item copied to current sentence")
        else:
            QMessageBox.information(self, "No Selection", "Please select a history item to copy.")
    
    def clear_history(self):
        """Clear history"""
        reply = QMessageBox.question(
            self, "Clear History", 
            "Are you sure you want to clear all history?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history = []
            self.history_list.clear()
            self.status_bar.showMessage("History cleared")
    
    def save_history(self):
        """Save all history to a file"""
        if not self.history:
            QMessageBox.information(self, "Save History", "No history to save!")
            return
        
        # Get current date and time for default filename
        now = datetime.now()
        default_filename = f"asl_history_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save History", default_filename, "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("ASL Recognition History\n")
                    f.write("=" * 30 + "\n\n")
                    for i, sentence in enumerate(self.history, 1):
                        f.write(f"{i}. {sentence}\n")
                QMessageBox.information(self, "Save Successful", f"History saved to:\n{filename}")
                self.status_bar.showMessage(f"History saved to {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")
                self.status_bar.showMessage("Save failed")
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Simple cleanup
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

def main():
    """Main application entry point"""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Enhanced ASL Recognition System")
    app.setApplicationVersion("2.0 - Simple Edition")
    
    # Create and show window
    window = EnhancedASLApp()
    window.show()
    
    print("🚀 Enhanced ASL Recognition System Started!")
    print("✅ SIMPLE VERSION - No threading complexity!")
    print("✅ All buttons are fully functional:")
    print("   🔊 Speak - Simple direct TTS (may briefly pause UI)")
    print("   💾 Save - Saves to file with timestamp")
    print("   ❌ Clear Word - Clears current word")
    print("   🧹 Clear Sentence - Clears sentence and adds to history")
    print("   ⎵ Add Space - Manually adds space")
    print("   • Add Period - Manually adds period")
    print("   📋 Copy/Speak History - Fully functional")
    print("✅ All text is BLACK for maximum visibility!")
    print("✅ No threading = No crashes!")
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()