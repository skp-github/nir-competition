__author__ = "Philipp Flotho"
"""
Copyright 2021 by Philipp Flotho, All rights reserved.
"""
import cv2
import numpy as np


class VideoWriter:
    def __init__(self, filepath, fps=10):
        self.writer = None
        self.filepath = filepath
        self.width = None
        self.height = None
        self.n_channels = None
        self.fps = fps

    def _check_range(self, min_val, max_val, checkrange):
        return

    def __call__(self, frame):
        if self.writer is None:
            self.height, self.width = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.filepath, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), self.fps,
                                          (self.width, self.height))
        if len(frame.shape) < 3:
            self.n_channels = 1
        else:
            self.n_channels = frame.shape[2]

        if self.n_channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if frame.dtype != np.uint8:
            min_val = np.min(frame)
            max_val = np.max(frame)
            if min_val > 0 and max_val < 1:
                frame = (255 * frame).astype(np.uint8)
            elif min_val > 0 and max_val < 255:
                frame = frame.astype(np.uint8)
            else:
                frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        self.writer.write(frame)

    def write_frames(self, frames):
        for frame in frames:
            self(frame)

    def __del__(self):
        if self.writer is not None:
            self.writer.release()


if __name__ == "__main__":
    """
    Creates noise as example:
    """
    video_writer = VideoWriter("test.mp4")
    for i in range(100):
        video_writer(np.random.rand(480, 640, 3))
