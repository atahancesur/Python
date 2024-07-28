#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install opencv-python


# In[1]:


import cv2

cap = cv2.VideoCapture("eye.mp4") #goz.mp4

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    frame = cv2.resize(frame, (800, 600))
    
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gri, (7, 7), 0)
    
    _,thresh = cv2.threshold(blur, 20,400, cv2.THRESH_BINARY_INV) #goruntuyu siyah beyaz arasinda degistirir
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    #print(contours)
    
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        #print(cnt)
        if area>100:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.line(frame, (x+int(w/2), 0), (x+int(h/2), frame.shape[0]), (0, 255, 0), 2)
            cv2.line(frame, (0, y+int(h/2)), (frame.shape[1], y+int(h/2)), (0, 255, 0), 2)
        
        
    
    cv2.imshow("eye", frame) #goz
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()


# In[ ]:




