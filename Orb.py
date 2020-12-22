import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import os.path
import time

prototypes = []
MAX_PROTOTYPES=20
lowe_ratio = 1.0
imgs = []
confidence_threshold = 0.7
durability = 0.01
Choice = True
LearnNewPrototypes = False
start_imgs_fns = []
start_imgs_fns.append("station2.jpg")
start_imgs_fns.append("ball.jpg")
start_imgs_fns.append("floor.jpg")
SZX=256
SZY=256
Size = (SZX,SZY) #the size in which we process the frame

def ReadImg(fname):
    im = cv.imread(fname, 0)
    img = cv.resize(im, Size)
    #print(img.dtype, img.shape, type(img))
    #smooth image
    #kernel = np.ones((3,3),np.float32)/9
    #img = cv.filter2D(img,-1,kernel)
    return img

start_imgs = []
for fn in start_imgs_fns:
    start_imgs.append(ReadImg(fn))

def getSubImage(rect, src):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int 
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv.warpAffine(src, M, src.shape[:2])
    tmp = cv.getRectSubPix(dst, size, center)
    return cv.resize(tmp, Size)

def LookAt(image, Learn=False, j=0):
    global prototypes, imgs
    finder = cv.ORB_create()
    kp2, des2 = finder.detectAndCompute(image,None)
    best_i = None
    best_confidence = 0
    best_good = [] #for viz
    #1. See which prototype matches best to the new image
    for i in range(len(prototypes)) if Choice else [j]:
        if i < len(prototypes) and not des2 is None and not (j>0 and Choice):
            (kp1, des1, useCount) = prototypes[i]
            if des1 is not None:
                #BFMatcher knn match
                bf = cv.BFMatcher()
                matches = bf.knnMatch(des1,des2, k=2)
                #Apply ratio test
                good = []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    m = pair[0]
                    n = pair[1]
                    if m.distance < lowe_ratio*n.distance:
                        good.append([m])
                #Calculate confidence
                if Learn:
                    prototypes[i] = (prototypes[i][0], prototypes[i][1], prototypes[i][2]-durability)
                confidence = len(good) / (max(len(des1), len(des2)))
                if confidence >= best_confidence and confidence > confidence_threshold:
                    best_i = i
                    best_confidence = confidence
                    #2. Increase priority of found prototype
                    if Learn:
                        prototypes[best_i] = (prototypes[best_i][0], prototypes[best_i][1], prototypes[best_i][2]+best_confidence)
                    best_good = good
    if Learn:
        prototypes.append((kp2, des2, 1))
        imgs.append(image)
        zipped_lists = zip(prototypes, imgs)
        sorted_pairs = sorted(zipped_lists, key=lambda x: x[0][2])
        tuples = zip(*sorted_pairs)
        prototypes, imgs = [list(tuple)[:MAX_PROTOTYPES] for tuple in tuples]
        #prototypes = sorted(prototypes, key=lambda x: x[2])[:MAX_PROTOTYPES]
        #TODO imgs would have to be sorted the same way!!
        
    return (best_i, best_confidence, best_good, kp2)

for img in start_imgs:
    LookAt(img, True)

def crop_bottom_half(image):
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = int(SZX*0.5), int(SZY)
    croppedImage = image[start_row:end_row , start_col:end_col]
    
    #height, width, channels = image.shape
    #cropped_img = image[image.shape[0]/2:image.shape[0]]
    return cv.resize(croppedImage, Size)
    #return image

lastFrame = None
while True:
    frame_file = "/tmp/frame.jpg"
    debug_file = "/tmp/debug.jpg"
    Debug = True
    if os.path.isfile(frame_file):
        os.remove(frame_file)
    os.system("sh Webcam.sh " + frame_file) #capture next frame (or use CV videocapture if working)
    imgFrame = crop_bottom_half(ReadImg(frame_file))
    if not lastFrame is None and LearnNewPrototypes:
        diffImage = cv.absdiff(imgFrame, lastFrame)
        debug_image = diffImage
    else:
        debug_image = imgFrame.copy()
    if not lastFrame is None and LearnNewPrototypes:
        
        mask = cv.inRange(diffImage, 30, 255)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
        cnts = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        best_area = 0
        best_contour = None
        for i in range(len(cnts)):
            c = cnts[i]
            area = cv.contourArea(c)
            if area > best_area:
                area = best_area
                best_contour = c
        if not best_contour is None:
            rect = cv.minAreaRect(best_contour)
            Cropped = getSubImage(rect, imgFrame)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(debug_image,[box],0,(255,255,255),2)
            LookAt(Cropped, True)
            #cv.drawContours(debug_image, [best_contour], 0, (255,255,255), 2)
    MUL=2 #increased res for display to draw prototypes at detected location directly
    debug_image = cv.resize(debug_image, (SZX*MUL,SZY*MUL))
    for i in range(len(imgs)):
        (best_i, best_confidence, best_good_matches, kp) = LookAt(imgFrame, False, i)
        #print(best_i, best_confidence)
        if best_i is None or len(best_good_matches) == 0:
            continue
        Pt = np.zeros(2)
        NumPt = 0.0
        for match in best_good_matches:
            idx1 = match[0].trainIdx
            Pt += kp[idx1].pt
            NumPt += 1.0
        Pt = Pt / NumPt
        s_img = cv.resize(imgs[best_i], (10*MUL,10*MUL))
        x_offset=int(Pt[0]*MUL)
        y_offset=int(Pt[1]*MUL)
        debug_image[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
        #cv.circle(debug_image, (int(Pt[0]), int(Pt[1])), 10, (255 if i==0 else 0,0,0), -1)
        
    lastFrame = imgFrame.copy()
    displayImg = debug_image
    #displayImg = cv.drawMatchesKnn(imgs[best_i], prototypes[best_i][0], debug_image, kp, best_good_matches, None, flags=2)
    if Debug:
        if os.path.isfile(debug_file):
            os.remove(debug_file)
        cv.imwrite(debug_file, displayImg)
        os.system("pkill gpicview")
        os.system("gpicview " + debug_file + " &")
        time.sleep(1.0)
        
