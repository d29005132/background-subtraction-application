import numpy as np
import webbrowser
import cv2
import pygame
import mediapipe as mp
import time


def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return ()
  return (x, y, w, h)

def putText(source, x, y, text, scale=2.5, color=(255,255,255)):
    org = (x,y)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = scale
    thickness = 5
    lineType = cv2.LINE_AA
    cv2.putText(source, text, org, fontFace, fontScale, color, thickness, lineType)
    
def main(cap):
    frame_count = 0
    previous_frame = None
    Rect_1_count = 0
    Rect_2_count = 0
    Rect_3_count = 0
    a = 0
    
    cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE);
    while(True):
        frame_count += 1
        
        ret, frame = cap.read()
        
        
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        w = frame_rgba.shape[1]
        h = frame_rgba.shape[0]
        white = 255 - np.zeros((h,w,4), dtype='uint8')
        
       
        cv2.rectangle(frame_rgba, (40, 440), (340, 640), (0, 0, 0), 5)
        cv2.rectangle(frame_rgba, (940, 40), (1240, 240), (0, 0, 0), 5)
        cv2.rectangle(frame_rgba, (1040, 540), (1240, 640), (0, 0, 0), 5)
        
        # 左邊方框內文字顯示
        if Rect_1_count<=1:
            cv2.putText(frame_rgba, 'open cv', (60, 500), cv2.FONT_HERSHEY_DUPLEX, 
                        1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame_rgba, 'opencv:{}%'.format(int((Rect_1_count/50)*100)), 
                        (60, 500), cv2.FONT_HERSHEY_DUPLEX, 
                        1, (0, 0, 0), 1, cv2.LINE_AA)
        # 右邊方框內文字顯示
        if Rect_2_count<=1:
            cv2.putText(frame_rgba, 'play music', (960, 100), cv2.FONT_HERSHEY_DUPLEX, 
                        1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame_rgba, 'play music:{}%'.format(int((Rect_2_count/50)*100)), 
                        (960, 100), cv2.FONT_HERSHEY_DUPLEX, 
                        1, (0, 0, 0), 1, cv2.LINE_AA)

        if Rect_3_count<=1:
            cv2.putText(frame_rgba, 'exit', (1060, 500), cv2.FONT_HERSHEY_DUPLEX, 
                        1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame_rgba, 'exit'.format(int((Rect_3_count/50)*100)), 
                        (1060, 500), cv2.FONT_HERSHEY_DUPLEX, 
                        1, (0, 0, 0), 1, cv2.LINE_AA)
 
        prepared_frame = cv2.cvtColor(frame_rgba, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)
        

        if (previous_frame is None):

            previous_frame = prepared_frame
            continue
        
 
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
        previous_frame = prepared_frame
        
        
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)
        
       
        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 5000:
               
                continue
   
            boundingRect_1 = intersection(cv2.boundingRect(contour), (40, 440, 300, 200))
            boundingRect_2 = intersection(cv2.boundingRect(contour), (940, 40, 300, 200))
            boundingRect_3 = intersection(cv2.boundingRect(contour), (1040, 540, 300, 200))
            if boundingRect_1:
                if (boundingRect_1[2] * boundingRect_1[3]) / (300 * 200) > 0.5:
                    cv2.rectangle(frame_rgba, (40, 440), (340, 640), (0, 0, 255), 2)
                    Rect_1_count += 1
                    
            if boundingRect_2:
                if (boundingRect_2[2] * boundingRect_2[3]) / (300 * 200) > 0.5:
                    cv2.rectangle(frame_rgba, (940, 40), (1240, 240), (0, 0, 255), 2)
                    Rect_2_count += 1

            if boundingRect_3:
                if (boundingRect_3[2] * boundingRect_3[3]) / (300 * 200) > 0.5:
                    cv2.rectangle(frame_rgba, (1040, 540), (1240, 640), (0, 0, 255), 2)
                    Rect_3_count += 1
           
        if Rect_1_count >= 50:
            webbrowser.open('https://opencv.org/')
            Rect_1_count = 0

        
        if Rect_2_count >= 50:
            pygame.mixer.init()
            pygame.mixer.music.load('light.mp3')
            pygame.mixer.music.play()
            Rect_2_count = 0

        if Rect_3_count >= 10:
            exit()
            Rect_3_count = 0
            
        if a == 0:
            output = frame_rgba.copy()
       
               
        
      
        cv2.imshow('HCI_hw4', output)
        
       
        if cv2.waitKey(1) == ord('e'):
            break

  
    cap.release()
    cv2.destroyAllWindows()
     
    

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    main(cap)
