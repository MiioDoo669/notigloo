import cv2
import mediapipe as mp
import time
import pygame
import os

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


video_path = "alarm.mp4"
temp_audio = "temp_audio.mp3"

if not os.path.exists(temp_audio):
    print("Extracting audio... Please wait.")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio)

pygame.mixer.init()
pygame.mixer.music.load(temp_audio)

cam = cv2.VideoCapture(0)
cap_v = cv2.VideoCapture(video_path)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

blink_start_time = 0
drowsy_limit = 1.5   
video_playing = False
drowsy_events = 0
blink_total = 0
is_blinking = False

while True:
    ret_c, frame = cam.read()
    if not ret_c: break
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Landmarks for Eye
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        eye_distance = left_eye_bottom.y - left_eye_top.y

        dot_color = (0, 255, 0)
        if eye_distance < 0.01:
            dot_color = (0, 0, 255)
            
        eye_indices = [159, 145, 133, 33, 7, 163, 144, 153, 154, 155]
        for idx in eye_indices:
            px = landmarks[idx]
            x, y = int(px.x * frame_w), int(px.y * frame_h)
            cv2.circle(frame, (x, y), 2, dot_color, -1)
            
        cv2.putText(frame, f"Dist: {eye_distance:.4f}", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

       
        if eye_distance < 0.01: 
            if not is_blinking:
                blink_total += 1
                is_blinking = True
        else:
            is_blinking = False

        
        if eye_distance < 0.01: 
            if blink_start_time == 0:
                blink_start_time = time.time()
            
            duration = time.time() - blink_start_time
            
            if duration > drowsy_limit:
              
                ret_v, v_frame = cap_v.read()
                if not ret_v:
                    cap_v.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_v, v_frame = cap_v.read()
                
                if ret_v:
                    
                    v_frame = cv2.resize(v_frame, (640, 360))
                    cv2.imshow('EMERGENCY ALARM VIDEO', v_frame)
                
                if not video_playing:
                    pygame.mixer.music.play(-1)
                    video_playing = True
                    drowsy_events += 1
        else:
            blink_start_time = 0
            if video_playing:
                try:
                    cv2.destroyWindow('EMERGENCY ALARM VIDEO')
                except: pass
                pygame.mixer.music.stop()
                video_playing = False

   
    
    cv2.putText(frame, f"Blinks: {blink_total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Drowsy Alerts: {drowsy_events}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    cv2.imshow('Safety Monitoring System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cap_v.release()
cv2.destroyAllWindows()