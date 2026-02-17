import cv2 as cv
import mediapipe as mp
import time
import math
class postDetector():
  def __init__(self,mode =False,model_complexity = False, smooth_landmarks = True, 
               min_detection_con = 0.5,min_tracking_con = 0.5):
    self.mode = mode
    self.model_complexity = model_complexity
    self.smooth_landmarks = smooth_landmarks
    self.min_detection_con = min_detection_con
    self.min_tracking_con = min_tracking_con
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose(static_image_mode = self.mode, 
                                 model_complexity = self.model_complexity,
                                 smooth_landmarks =self.smooth_landmarks,
                                 min_detection_confidence=self.min_detection_con,
                                 min_tracking_confidence=self.min_tracking_con)
    self.mpDraw = mp.solutions.drawing_utils
    self.results = None


  def findPose(self, img, draw = True):
    self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    self.results = self.pose.process(self.imgRGB)

    if self.results.pose_landmarks:
      if draw:
        self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
    return img

  def findPosition(self,img,draw=True):
    self.lmList = []
    if self.results.pose_landmarks:
      for id, lm in enumerate(self.results.pose_landmarks.landmark):
          h,w,c = img.shape
          cx,cy = int(lm.x *w), int(lm.y*h)
          self.lmList.append([id,cx,cy])
          if draw:
            cv.circle(img,(cx,cy),10,(255,0,0),cv.FILLED) 
    return self.lmList
  
  def findAngle(self,img,p1,p2,p3,draw = True):
    #Get the landmarks
    x1, y1 = self.lmList[p1][1:]
    x2, y2 = self.lmList[p2][1:]
    x3, y3 = self.lmList[p3][1:]

    #Calculate the Angle
    angle = math.degrees(math.atan2(y3-y2,x3-x2)- math.atan2(y1-y2,x1-x2))
    #print(angle)

    if angle < 0:
      angle += 360


    if draw:
      cv.line(img,(x1,y1),(x2,y2),(255,255,255),3)
      cv.line(img,(x3,y3),(x2,y2),(255,255,255),3)
      cv.circle(img,(x1,y1),10,(0,0,255),cv.FILLED)
      cv.circle(img,(x1,y1),15,(0,0,255),2)
      cv.circle(img,(x2,y2),10,(0,0,255),cv.FILLED) 
      cv.circle(img,(x2,y2),15,(0,0,255),2)
      cv.circle(img,(x3,y3),10,(0,0,255),cv.FILLED) 
      cv.circle(img,(x3,y3),15,(0,0,255),2)
      cv.putText(img,str(int(angle)),(x2 - 50,y2 + 50),
                 cv.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    return angle
def main():
  cap = cv.VideoCapture('Videos/PoseEstimationVid5.mp4')
  pTime = 0
  detector = postDetector()

  while True:
    success, img = cap.read()

    img = cv.resize(img,(960,512),interpolation=cv.INTER_AREA)
    img = detector.findPose(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
      print(lmList)
      cv.circle(img,(lmList[14][1], lmList[14][2]), 5, (255,0,0),cv.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_TRIPLEX, 2, 
              (255,255,255),2)

    cv.imshow("Image",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv.destroyAllWindows()


if __name__ == "__main__":
    main()