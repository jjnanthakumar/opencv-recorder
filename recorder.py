import numpy as np
import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
from datetime import datetime
from win32api import GetSystemMetrics
from PIL import ImageGrab,Image

class VideoRecorder():  
    "Video class based on openCV"
    __Fname = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.mp4"
    def __init__(self, name="temp_video.mp4", fourcc="mp4v", camindex=0, fps=20):
        w, h = GetSystemMetrics(0), GetSystemMetrics(1)
        self.open = True
        self.device_index = camindex
        self.fps = fps                  # fps should be the minimum constant rate at which the camera can
        self.fourcc = fourcc            # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (w, h) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = name
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()

    def rect_with_rounded_corners(self,image, r, t, c):
        """
        :param image: image as NumPy array
        :param r: radius of rounded corners
        :param t: thickness of border
        :param c: color of border
        :return: new image as NumPy array with rounded corners
        """

        c += (255, )

        h, w = image.shape[:2]

        # Create new image (three-channel hardcoded here...)
        new_image = np.ones((h+2*t, w+2*t, 4), np.uint8) * 255
        new_image[:, :, 3] = 0

        # Draw four rounded corners
        new_image = cv2.ellipse(new_image, (int(r+t/2), int(r+t/2)), (r, r), 180, 0, 90, c, t)
        new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(r+t/2)), (r, r), 270, 0, 90, c, t)
        new_image = cv2.ellipse(new_image, (int(r+t/2), int(h-r+3*t/2-1)), (r, r), 90, 0, 90, c, t)
        new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(h-r+3*t/2-1)), (r, r), 0, 0, 90, c, t)

        # Draw four edges
        new_image = cv2.line(new_image, (int(r+t/2), int(t/2)), (int(w-r+3*t/2-1), int(t/2)), c, t)
        new_image = cv2.line(new_image, (int(t/2), int(r+t/2)), (int(t/2), int(h-r+3*t/2)), c, t)
        new_image = cv2.line(new_image, (int(r+t/2), int(h+3*t/2)), (int(w-r+3*t/2-1), int(h+3*t/2)), c, t)
        new_image = cv2.line(new_image, (int(w+3*t/2), int(r+t/2)), (int(w+3*t/2), int(h-r+3*t/2)), c, t)

        # Generate masks for proper blending
        mask = new_image[:, :, 3].copy()
        mask = cv2.floodFill(mask, None, (int(w/2+t), int(h/2+t)), 128)[1]
        mask[mask != 128] = 0
        mask[mask == 128] = 1
        mask = np.stack((mask, mask, mask), axis=2)

        # Blend images
        temp = np.zeros_like(new_image[:, :, :3])
        temp[(t-1):(h+t-1), (t-1):(w+t-1)] = image.copy()
        new_image[:, :, :3] = new_image[:, :, :3] * (1 - mask) + temp * mask

        # Set proper alpha channel in new image
        temp = new_image[:, :, 3].copy()
        new_image[:, :, 3] = cv2.floodFill(temp, None, (int(w/2+t), int(h/2+t)), 255)[1]

        return new_image

    def record(self):
        "Video starts being recorded"
        while self.open:
            img = ImageGrab.grab(bbox=(0, 0, self.frameSize[0], self.frameSize[1]))
            ret, frame = self.video_cap.read()
            frame = cv2.resize(frame,(150,150),interpolation=cv2.INTER_AREA)
            frame1 = self.rect_with_rounded_corners(np.array(frame),50,1,(0,0,255))
            frame1 = cv2.cvtColor(cv2.cvtColor(np.array(frame1),cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2RGB)
            real = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            fr_height, fr_width, _ = frame1.shape
            real[:fr_height, :fr_width, :] = frame1[:fr_height, :fr_width, :]
            cv2.imshow("Secret Capture", real)
            if ret:
                self.frame_counts += 1
                time.sleep(1/self.fps)
                self.video_out.write(real)
                print(self.frame_counts)
                if cv2.waitKey(10) == 27:
                    break
            else:
                break

    def stop(self):
        "Finishes the video recording therefore the thread too"
        if self.open:
            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

    def start(self):
        "Launches the video recording function using a thread"
        video_thread = threading.Thread(target=self.record)
        video_thread.start()

class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self, filename="temp_audio.wav", rate=44100, fpb=1024, channels=2):
        self.open = True
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio_filename = filename
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []

    def record(self):
        "Audio starts being recorded"
        self.stream.start_stream()
        while self.open:
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if not self.open:
                break

    def stop(self):
        "Finishes the audio recording therefore the thread too"
        if self.open:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

    def start(self):
        "Launches the audio recording function using a thread"
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

def start_AVrecording(filename="test"):
    global video_thread
    global audio_thread
    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()
    audio_thread.start()
    video_thread.start()
    return filename

def start_video_recording(filename="test"):
    global video_thread
    video_thread = VideoRecorder()
    video_thread.start()
    time.sleep(30)
    video_thread.stop()
    return filename

def start_audio_recording(filename="test"):
    global audio_thread
    audio_thread = AudioRecorder()
    audio_thread.start()
    return filename
def stop_AVrecording(filename="test"):
    audio_thread.stop() 
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop() 

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)

    # Merging audio and video signal
    if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
        print("Re-encoding")
        # cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
        # subprocess.call(cmd, shell=True)
        print("Muxing")
        cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.mp4 -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)
    else:
        print("Normal recording\nMuxing")
        cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.mp4 -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)
        print("..")

def file_manager(filename="test"):
    "Required and wanted processing of final files"
    local_path = os.getcwd()
    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")
    if os.path.exists(str(local_path) + "/temp_video.avi"):
        os.remove(str(local_path) + "/temp_video.avi")
    if os.path.exists(str(local_path) + "/temp_video2.avi"):
        os.remove(str(local_path) + "/temp_video2.avi")
    # if os.path.exists(str(local_path) + "/" + filename + ".avi"):
    #     os.remove(str(local_path) + "/" + filename + ".avi")

start_video_recording()

# start_AVrecording('temp')
# time.sleep(30)
# stop_AVrecording('temp')

# %%
