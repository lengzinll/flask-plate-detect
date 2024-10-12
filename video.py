import cv2
from threading import Thread

class VideoStream:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.ret, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.capture.read()
            if not self.ret:
                self.stop()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()