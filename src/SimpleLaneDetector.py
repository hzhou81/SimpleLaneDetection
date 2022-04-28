import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def filter_only_yellow_white(img):
    hlsColorspacedImage=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    yellowLower=np.array([10,0,90])
    yellowUpper=np.array([50,255,255])
    yellowMask=cv2.inRange(hlsColorspacedImage,yellowLower,yellowUpper)
    whiteLower=np.array([0,190,10])
    whiteUpper=np.array([255,255,255])
    whiteMask=cv2.inRange(hlsColorspacedImage,whiteLower,whiteUpper)
    mask=cv2.bitwise_or(yellowMask,whiteMask)
    lineImg=cv2.bitwise_and(img,img,mask=mask)
    return lineImg

def crop_region_of_interest(img):
    shapes = np.array([[(0, img.shape[0]), (int(img.shape[1]*0.45), int(img.shape[0]*0.645)), (int(img.shape[1]*0.55), int(img.shape[0]*0.645)), (img.shape[1], img.shape[0])]])
    filledPolygon=np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        color_to_filter = (255,) * channel_count
    else:
        color_to_filter = 255
    #print(shapes)
    cv2.fillPoly(filledPolygon, shapes, color_to_filter)
    masked_img = cv2.bitwise_and(img, filledPolygon)
    return masked_img

def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #edges = cv2.Canny(gray, 50, 120)
    return gray

def draw_lanes(img,lines, color=[255, 0, 0], thickness=2):
    rightSlope=[]
    rightIntercept=[]
    leftSlope=[]
    leftIntercept=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            yDiff=y1-y2
            xDiff=x1-x2
            slope=yDiff/xDiff
            yIntecept=y2-(slope*x2)
            if slope>0.3 and x1>int(0.4*img.shape[1]):
                rightSlope.append(slope)
                rightIntercept.append(yIntecept)
            elif slope<-0.3 and x1<(0.6*img.shape[1]):
                leftSlope.append(slope)
                leftIntercept.append(yIntecept)
    if len(leftSlope)==0: return
    if len(rightSlope)==0: return
    leftAvgSlope=sum(leftSlope)/len(leftSlope)
    leftAvgIntercept=sum(leftIntercept)/len(leftIntercept)
    rightAvgSlope=sum(rightSlope)/len(rightSlope)
    rightAvgIntercept=sum(rightIntercept)/len(rightIntercept)
    leftLineX1=int((0.645*img.shape[0]-leftAvgIntercept)/leftAvgSlope)
    leftLineX2=int((img.shape[0]-leftAvgIntercept)/leftAvgSlope)
    rightLineX1=int((0.645*img.shape[0]-rightAvgIntercept)/rightAvgSlope)
    rightLineX2=int((img.shape[0]-rightAvgIntercept)/rightAvgSlope)
    cv2.line(img,(leftLineX1,int(0.645*img.shape[0])),(leftLineX2,img.shape[0]),color,thickness)
    cv2.line(img,(rightLineX1,int(0.645*img.shape[0])),(rightLineX2,img.shape[0]),color,thickness)

def detect_lane(img):
    colorFilteredImage=filter_only_yellow_white(img)
    regionOfInterest=crop_region_of_interest(colorFilteredImage)
    edgesOnly=detect_edges(regionOfInterest)
    lines = cv2.HoughLinesP(edgesOnly, 1, np.pi / 180, 10, np.array([]), minLineLength=20,maxLineGap=100)
    draw_lanes(img, lines)
    return img

output = '../resources/video_3_sol.mp4'
clip = VideoFileClip("../resources/video_3.mp4")
out_clip = clip.fl_image(detect_lane)
out_clip.write_videofile(output, audio=False)
