import cv2 as cv
import mediapipe as mp
import numpy as np
import time

class FaceDetector():
    def __init__(self, minDectectionCon=0.5):
        self.minDectectionCon = minDectectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDectectionCon)

    def findFaces(self,img, draw = True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []

        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                
                    cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_COMPLEX, 1, (100,0,255), thickness=1)
        return img, bboxs
    
    def fancyDraw(self, img, bbox, l = 30, t=3):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv.rectangle(img, bbox,(255,250,0), 1)
        # Top Left x,y
        cv.line(img, (x,y), (x+l,y), (255,250,0), t)
        cv.line(img, (x,y), (x, y+l), (255,250,0), t)
        # Top Right x,y1
        cv.line(img, (x1,y), (x1-l,y), (255,250,0), t)
        cv.line(img, (x1,y), (x1, y+l), (255,250,0), t)
        # Bottom Right
        cv.line(img, (x,y1), (x+l,y1), (255,250,0), t)
        cv.line(img, (x,y1), (x, y1-l), (255,250,0), t)
        # Bottom Left
        cv.line(img, (x1,y1), (x1-l,y1), (255,250,0), t)
        cv.line(img, (x1,y1), (x1, y1-l), (255,250,0), t)

        return img


def main():
    cap = cv.VideoCapture('Video.mp4')
    pTime = 1
    detector = FaceDetector()
    while True:
        success, img = cap.read()

        if not success or img is None:
            break  # Break the loop if there's an issue with reading the frame

        img, bboxs = detector.findFaces(img)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Update the following lines to ensure img is a valid numpy array before using cv.putText
        if isinstance(img, np.ndarray):  
            cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), thickness=3)
            cv.imshow("Video", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break  # Break the loop if 'q' key is pressed

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()