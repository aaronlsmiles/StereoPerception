import numpy as np
import cv2
from tqdm import tqdm
import os
from matplotlib import pyplot as plt


# Set the path to the images captured by the left and right cameras
pathL = "./data/Y/frameL/"
pathR = "./data/Y/frameR/"

print("Extracting image coordinates of respective 3D pattern....\n")

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(30, 100)):
    imgL = cv2.imread(pathL + "img%d.png" % i)
    imgR = cv2.imread(pathR + "img%d.png" % i)
    imgL_gray = cv2.imread(pathL + "img%d.png" % i, 0)
    imgR_gray = cv2.imread(pathR + "img%d.png" % i, 0)
    outputL = imgL.copy()
    outputR = imgR.copy()
    retR, cornersR = cv2.findChessboardCorners(outputR, (7, 6), None)
    retL, cornersL = cv2.findChessboardCorners(outputL, (7, 6), None)

    if retR and retL:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(outputR, (7, 6), cornersR, retR)
        cv2.drawChessboardCorners(outputL, (7, 6), cornersL, retL)
        cv2.imshow('cornersR', outputR)
        #  cv2.circle(outputR, (int(cornersR[0][0][0]), int(cornersR[0][0][1])), 5, (0, 0, 255), -1)
        cv2.imshow('cornersL', outputL)
        #  cv2.circle(outputL, (int(cornersL[0][0][0]), int(cornersL[0][0][1])), 5, (0, 0, 255), -1)
        cv2.waitKey(0)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)

print("Calculating left camera parameters... ")
# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

mean_error = 0
for i in range(len(obj_pts)):
    imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecsL[i], tvecsL[i], mtxL, distL)
    error = cv2.norm(img_ptsL[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error (left): {}".format(mean_error / len(obj_pts)))

#  draw imgpoints on image to check calibration accuracy
#  Lcircles = cv2.circle(imgL, (int(imgpoints2[0][0][0]), int(imgpoints2[0][0][1])), 3, (0, 0, 255), -1)
#  cv2.imshow('Lcircles', Lcircles)

print("Calculating right camera parameters ... ")
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

mean_error = 0
for i in range(len(obj_pts)):
    imgpoints3, _ = cv2.projectPoints(obj_pts[i], rvecsR[i], tvecsR[i], mtxR, distR)
    error = cv2.norm(img_ptsR[i], imgpoints3, cv2.NORM_L2) / len(imgpoints3)
    mean_error += error
print("total error (right): {}".format(mean_error / len(obj_pts)))

#  draw imgpoints on image to check calibration accuracy
#  Rcircles = cv2.circle(imgR, (int(imgpoints3[0][0][0]), int(imgpoints3[0][0][1])), 3, (0, 0, 255), -1)
#  cv2.imshow('Rcircles', Rcircles)

print("Stereo calibration.....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrices so that only Rot, Trns, Emat and Fmat are calculated.
# Hence, intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                                                    img_ptsL,
                                                                                    img_ptsR,
                                                                                    new_mtxL,
                                                                                    distL,
                                                                                    new_mtxR,
                                                                                    distR,
                                                                                    imgL_gray.shape[::-1],
                                                                                    criteria_stereo,
                                                                                    flags)

# Save the fundamental matrix
np.save('data/Y/Fmat.npy', Fmat)

# Once we know the transformation between the two cameras we can perform stereo rectification.
# StereoRectify function
rectify_scale = 1  # if 0 image cropped, if 1 image not cropped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                          imgL_gray.shape[::-1], Rot, Trns,
                                                                          rectify_scale, (0, 0))

# Use the rotation matrices for stereo rectification and camera intrinsics for un-distorting the image
# Compute the rectification map (mapping between the original image pixels and 
# their transformed values after applying rectification and un-distortion) for left and right camera frames
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               imgR_gray.shape[::-1], cv2.CV_16SC2)

save_directory = r'C:\Users\joini\OneDrive\Documents\code\DCE\stereoVision\StereoPerception\depth-perception\data\Y'
os.chdir(save_directory)
print("Before saving files:")
print(os.listdir(save_directory))

print("Saving parameters ......")
cv_file = cv2.FileStorage("params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.release()

# print images before and after rectification
cv2.imshow("Left image before rectification", imgL)
cv2.imshow("Right image before rectification", imgR)

Left_nice = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
Right_nice = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

cv2.imshow("Left image after rectification", Left_nice)
cv2.imshow("Right image after rectification", Right_nice)
cv2.imwrite('Left_rect.png', Left_nice)
cv2.imwrite('Right_rect.png', Right_nice)
print("saving rectified images:")
print(os.listdir(save_directory))

print('Successfully saved')
cv2.waitKey(0)




############################################################################################################
# Find keypoints and draw eplines on Left_nice and Right_nice images

img1 = Left_nice
img2 = Right_nice

# 1. Detect keypoints and their descriptors
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Visualize keypoints
imgSift = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("SIFT Keypoints", imgSift)

# Match keypoints in both images
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


# Draw the keypoint matches between both pictures
# Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask[300:500],
                   flags=cv2.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv2.drawMatchesKnn(
    img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
cv2.imshow("Keypoint matches", keypoint_matches)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Visualize epilines
# From: https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/#zp-ID-2272-3683967-MB926RH5
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, *_ = img1src.shape
    img1color = img1src
    img2color = img2src
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, Fmat)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, Fmat)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.suptitle("Epilines in both images")
plt.show()
