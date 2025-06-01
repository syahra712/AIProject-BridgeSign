import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from symspellpy.symspellpy import SymSpell, Verbosity
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import pyttsx3
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ======= Load Model =======
model_dict = pickle.load(open("/Users/admin/Desktop/American-Sign-language-Detection-System/modelbest.p", 'rb'))
model = model_dict['model']
expected_features = 42

# ======= SymSpell Setup =======
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("/Users/admin/Desktop/American-Sign-language-Detection-System/dictionary.txt", term_index=0, count_index=1)

# ======= Mediapipe Setup =======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ======= Label Dictionary =======
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict.update({26 + i: str(i) for i in range(10)})
labels_dict[36] = ' '
labels_dict[37] = '.'

# ======= Text to Speech =======
engine = pyttsx3.init()

# ======= PyQt GUI App =======
class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Recognition with Suggestions")
        self.setGeometry(100, 100, 1000, 600)

        # Buffers
        self.buffer = []
        self.stable_char = None
        self.word_buffer = ""
        self.sentence = ""
        self.last_time = time.time()
        self.registration_delay = 2.5

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # UI Elements
        self.video_label = QLabel()
        self.char_label = QLabel("Current Letter: ---")
        self.word_label = QLabel("Current Word: ")
        self.sent_label = QLabel("Sentence: ")

        self.speak_btn = QPushButton("🔊 Speak")
        self.clear_word_btn = QPushButton("❌ Clear Word")
        self.clear_sent_btn = QPushButton("🧹 Clear Sentence")

        self.suggest_buttons = [QPushButton() for _ in range(5)]
        for btn in self.suggest_buttons:
            btn.clicked.connect(self.use_suggestion)

        # Layouts
        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label)
        vbox.addWidget(self.char_label)
        vbox.addWidget(self.word_label)
        vbox.addWidget(self.sent_label)

        h_suggestions = QHBoxLayout()
        for btn in self.suggest_buttons:
            h_suggestions.addWidget(btn)

        h_actions = QHBoxLayout()
        h_actions.addWidget(self.speak_btn)
        h_actions.addWidget(self.clear_word_btn)
        h_actions.addWidget(self.clear_sent_btn)

        vbox.addLayout(h_suggestions)
        vbox.addLayout(h_actions)
        self.setLayout(vbox)

        # Button connections
        self.speak_btn.clicked.connect(self.speak)
        self.clear_word_btn.clicked.connect(self.clear_word)
        self.clear_sent_btn.clicked.connect(self.clear_sentence)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        output = frame.copy()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_, y_ = [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - min(x_), lm.y - min(y_)])

                if len(data_aux) < expected_features:
                    data_aux.extend([0] * (expected_features - len(data_aux)))
                elif len(data_aux) > expected_features:
                    data_aux = data_aux[:expected_features]

                prediction = model.predict([np.asarray(data_aux)])
                predicted_char = labels_dict[int(prediction[0])]

                self.buffer.append(predicted_char)
                if len(self.buffer) > 30:
                    self.buffer.pop(0)

                if self.buffer.count(predicted_char) > 25:
                    current_time = time.time()
                    if current_time - self.last_time > self.registration_delay:
                        self.stable_char = predicted_char
                        self.last_time = current_time

                        if predicted_char == ' ':
                            if self.word_buffer:
                                self.sentence += self.word_buffer + " "
                                self.word_buffer = ""
                        elif predicted_char == '.':
                            if self.word_buffer:
                                self.sentence += self.word_buffer + "."
                                self.word_buffer = ""
                        else:
                            self.word_buffer += predicted_char

        # Update UI
        self.char_label.setText(f"Current Letter: {self.stable_char if self.stable_char else '---'}")
        self.word_label.setText(f"Current Word: {self.word_buffer}")
        self.sent_label.setText(f"Sentence: {self.sentence}")

        # Word suggestions
        suggestions = sym_spell.lookup(self.word_buffer.lower(), Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
        for i in range(5):
            if i < len(suggestions):
                self.suggest_buttons[i].setText(suggestions[i].term)
                self.suggest_buttons[i].setEnabled(True)
            else:
                self.suggest_buttons[i].setText("")
                self.suggest_buttons[i].setEnabled(False)

        # Convert to QImage
        img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        qt_img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def use_suggestion(self):
        btn = self.sender()
        selected_word = btn.text()
        self.sentence += selected_word + " "
        self.word_buffer = ""

    def speak(self):
        if self.sentence.strip():
            engine.say(self.sentence.strip())
            engine.runAndWait()

    def clear_word(self):
        self.word_buffer = ""

    def clear_sentence(self):
        self.sentence = ""

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLApp()
    window.show()
    sys.exit(app.exec_())
