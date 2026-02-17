import cv2 as cv
import mediapipe as mp
import time
import math

class handDetector():
  def __init__(self,mode = False,maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(
                                    static_image_mode=self.mode,
                                    max_num_hands=self.maxHands,
                                    min_detection_confidence=self.detectionCon,
                                    min_tracking_confidence=self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils
    self.lmList = []
    self.tipIds = [4,8,12,16,20]
    self.hand_types = []

  def findHands(self,img, draw = True):
    self.imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    self.result = self.hands.process(self.imgRGB)
    self.hand_types = []   # reset each frame

    #print(result.multi_hand_landmarks)
    if self.result.multi_hand_landmarks:

      for handLMs, handType in zip(self.result.multi_hand_landmarks,self.result.multi_handedness):        
        self.mpDraw.draw_landmarks(img, landmark_list = handLMs,connections = self.mpHands.HAND_CONNECTIONS)
        
        # #Draw text on the hand
        # w,h,c = img.shape
        # xList = [int(lm.x * w) for lm in handLMs.landmark]
        # yList = [int(lm.y * h) for lm in handLMs.landmark]
        # xmin, xmax = min(xList), max(xList)
        # ymin, ymax = min(yList), max(yList)
        # cx,cy = (xmax - xmin)//2,(ymax-ymin)//2
        self.hand_types.append(handType.classification[0].label)

    return img
  
  def findPosition(self,img,handNo=0,draw=True):
    xList = []
    yList = []
    bbox = []
    self.lmList = []
    if self.result.multi_hand_landmarks:
      myHand = self.result.multi_hand_landmarks[handNo]

      for id, lm in enumerate(myHand.landmark):

        #print(id,lm)
        h, w, c = img.shape
        cx, cy = int(lm.x * w ), int(lm.y * h)
        xList.append(cx)
        yList.append(cy)
        #print(id,cx,cy)

        self.lmList.append([id, cx, cy])

        if draw:
          cv.circle(img,(cx,cy), 5 , (9,255,0), cv.FILLED)
    
      xmin,xmax = min(xList), max(xList)
      ymin,ymax = min(yList), max(yList)
      bbox = xmin,ymin,xmax,ymax

      if draw:
        cv.rectangle(img,(xmin - 20,ymin - 20),(xmax + 20,ymax + 20),
                    (0,255,0),2)
      
    return self.lmList,bbox
  
  def fingersUp(self,hand = 'Right'):
    fingers = []

    if not self.lmList or len(self.lmList) < 21:
      return [0,0,0,0,0]
    

    #Thumb
    if hand == 'Right':
        fingers.append(1 if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1] else 0)
    else:
        fingers.append(1 if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1] else 0)

    #4 fingers
    for id in range(1,5):
        fingers.append(1 if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2] else 0)
      
    return fingers
      
  def findDistance(self, p1,p2,img,draw=True,r=15,t=3):
    x1,y1 = self.lmList[p1][1:]
    x2,y2 = self.lmList[p2][1:]
    cx,cy = (x1+x2)//2,(y1+y2)//2

    if draw:
      cv.line(img,(x1,y1),(x2,y2),(0,255,255),t)
      cv.circle(img,(x1,y1),r,(0,255,255),cv.FILLED)
      cv.circle(img,(x2,y2),r,(0,255,255),cv.FILLED)
      cv.circle(img,(cx,cy),r,(0,0,255),cv.FILLED)
    length = math.hypot(x2-x1,y2-y1)
    return length,img,[x1,y1,x2,y2,cx,cy]


def main():

  cap = cv.VideoCapture(0)
  pTime = 0
  cTime = 0
  detector = handDetector()
  while True:
    success, img = cap.read()

    img = cv.flip(img,1)   
    img = detector.findHands(img,draw=False)
    lmList,bbox = detector.findPosition(img,draw=False)

    cTime = time.time()
    fps= 1/(cTime - pTime)
    pTime = cTime
    
    cv.putText(img,str(int(fps)), (10,70),cv.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), 3)
    cv.imshow("Image", img)

    key = cv.waitKey(1) & 0xFF

    if key in [ord('d'),ord('D')]:
      break
  
  cap.release()
  cv.destroyAllWindows()
if __name__ == '__main__':
    main()