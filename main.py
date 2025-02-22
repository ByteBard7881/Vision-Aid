import cv2
import threading
import time
import pyttsx3
from ultralytics import YOLO
import queue
import os

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform


class YoloApp(App):
    def build(self):
        if platform == "android":
            from android.storage import app_storage_path

            model_path = os.path.join(app_storage_path(), "best.pt")
        else:
            model_path = "best.pt"

        self.layout = BoxLayout()
        self.image = Image()
        self.layout.add_widget(self.image)

        self.model = YOLO("best.pt")
        self.capture = cv2.VideoCapture(0)

        self.REAL_WIDTH = 20
        self.KNOWN_DISTANCE = 100
        self.KNOWN_WIDTH_IN_PIXELS = 150
        self.FOCAL_LENGTH = (
            self.KNOWN_WIDTH_IN_PIXELS * self.KNOWN_DISTANCE
        ) / self.REAL_WIDTH

        self.speech_queue = queue.Queue()
        self.last_alert_time = 0
        self.ALERT_COOLDOWN = 3

        tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        tts_thread.start()

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    def tts_worker(self):
        engine = pyttsx3.init()
        while True:
            message = self.speech_queue.get()
            if message is None:
                break
            engine.say(message)
            engine.runAndWait()
            self.speech_queue.task_done()

    def estimate_distance(self, bbox_width):
        if bbox_width > 0:
            return (self.REAL_WIDTH * self.FOCAL_LENGTH) / bbox_width
        return -1

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        results = self.model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                distance = self.estimate_distance(width)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{distance:.2f} cm",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if distance < 60:
                    current_time = time.time()
                    if (current_time - self.last_alert_time) > self.ALERT_COOLDOWN:
                        predicted_class = self.model.names[int(box.cls)]
                        self.last_alert_time = current_time
                        self.speech_queue.put(predicted_class)

        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.image.texture = texture

    def on_stop(self):
        self.capture.release()
        self.speech_queue.put(None)


if __name__ == "__main__":
    YoloApp().run()
