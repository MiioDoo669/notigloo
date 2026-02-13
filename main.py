import cv2
import mediapipe as mp
import time
import pygame
from moviepy.editor import VideoFileClip # Audio extract လုပ်ရန်
import os

# --- ၁။ Video ထဲက အသံကို Extract လုပ်ခြင်း ---
video_path = "alarm.mp4"
temp_audio = "temp_audio.mp3"

if not os.path.exists(temp_audio):
    print("Extracting audio from video... Please wait.")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio)

# --- ၂။ Pygame Setup ---
pygame.mixer.init()
pygame.mixer.music.load(temp_audio)

# Camera & Video Setup
cam = cv2.VideoCapture(0)
cap_v = cv2.VideoCapture(video_path)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

blink_start_time = 0
drowsy_limit = 2.0  
video_playing = False

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        eye_distance = left_eye_bottom.y - left_eye_top.y

        if eye_distance < 0.007:
            if blink_start_time == 0:
                blink_start_time = time.time()
            
            blink_duration = time.time() - blink_start_time
            
            if blink_duration > drowsy_limit:
                # Video Frame ဖတ်ခြင်း
                ret, v_frame = cap_v.read()
                if not ret:
                    cap_v.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, v_frame = cap_v.read()
                
                # Window အသစ်ဖြင့် Video ပြခြင်း
                cv2.imshow('EMERGENCY ALARM VIDEO', v_frame)
                
                if not video_playing:
                    pygame.mixer.music.play(-1)
                    video_playing = True
        else:
            blink_start_time = 0
            if video_playing:
                cv2.destroyWindow('EMERGENCY ALARM VIDEO')
                pygame.mixer.music.stop()
                video_playing = False

    cv2.imshow('Main Camera View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cap_v.release()
cv2.destroyAllWindows()