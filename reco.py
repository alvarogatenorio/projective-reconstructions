import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
import pptk

# Loads two images.
def load_images(name1, name2):
    img1 = cv.imread(name1, 0)
    img2 = cv.imread(name2, 0)
    return img1, img2

'''
Computes keypoints of one image using the SURF algorithm. Based on:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
'''
def compute_keypoints(img, threshold):
    # Creates a SURF object with the given hessian threshold.
    surf = cv.xfeatures2d.SURF_create(threshold)
    # Compute key points and descriptors of the image.
    keypoints, descriptors = surf.detectAndCompute(img, None)
    # Saving results.
    output = cv.drawKeypoints(img, keypoints, None, (255, 0, 0), 4)
    plt.imsave('key.png', output)
    return keypoints, descriptors


'''
Computes keypoints of two images using the SURF algorithm. Based on:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
'''
def compute_keypoints(img1, img2, threshold):
    # Creates a SURF object with the given hessian threshold.
    surf = cv.xfeatures2d.SURF_create(threshold)
    # Compute key points and descriptors of each image.
    keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2, None)
    # Saving results.
    output1 = cv.drawKeypoints(img1, keypoints1, None, (255,0,0), 4)
    output2 = cv.drawKeypoints(img2, keypoints2, None, (255,0,0), 4)
    plt.imsave('key1.png', output1)
    plt.imsave('key2.png', output2)
    return keypoints1, descriptors1, keypoints2, descriptors2

'''
Computes correspondences of keypoints of two images using Lowe's rule. Based on:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

Returns two lists with the coordinates of the corresponding points.
'''
def compute_correspondences(descriptors1, descriptors2, keypoints1, keypoints2, img1 = [], img2 = []):
    # Creates a brute force matcher object.
    matcher = cv.BFMatcher()
    '''
    Matches each descriptor with its 2 nearest neighbours.
    Returns a list of list of 2 DMatch objects sorted by distance.
    Each descriptor in descriptors1 is called "query descriptor".
    Each descriptor in descriptors2 is called "train descriptor".
    '''
    matches_lowe = matcher.knnMatch(descriptors1, descriptors2, k = 2)

    # Invariant: img1_points[i] is a matching point with img2_ponts[i].
    img1_points = []
    img2_points = []
    good_matches = []
    # In each iteration selects the two DMatch objects corresponding to the same query descriptor.
    for match1, match2 in matches_lowe:
        # Applying Lowe's rule with 0.8 threshold (match1 is the nearest match).
        if match1.distance < 0.8 * match2.distance:
            '''
            trainidx returns the index associated to match1 in descriptors1.
            queryidx resturns the index associated to match2 in descriptors2.
            pt returns the coordinates of the keypoint.
            '''
            good_matches.append([match1])
            img1_points.append(keypoints1[match1.queryIdx].pt)
            img2_points.append(keypoints2[match1.trainIdx].pt)
    # Saving results.
    if img1 != [] and img2 != []:
        output = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags = 2)
        plt.imsave('matches.png', output)
    # Casting data to int32.
    return np.int32(img1_points), np.int32(img2_points)

'''
Computes correspondences of keypoints of two images using brute force or cross check. Based on:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''
def compute_correspondences_brute_force(descriptors1, descriptors2, keypoints1, keypoints2, img1 = [], img2 = [], crossCheck = True):
    matcher = cv.BFMatcher(crossCheck = crossCheck)
    # Returns a vector of DMatch objects.
    matches = matcher.match(descriptors1, descriptors2)
    # Sorting by the distance of each match.
    matches = sorted(matches, key = lambda match: match.distance)
    # Saving results.
    if img1 != [] and img2 != []:
        output = cv.drawMatches(img1, key_points1, img2, key_points2, matches, None)
        plt.imsave('matches.png', output)
    return None

'''
Computes the fundamental matrix. Based on:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html#epipolar-geometry
and
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

Returns the fundamental matrix as well as the inlier corresponding points in each image and the epipoles.
'''
def compute_fundamental_matrix(img1_points, img2_points, img1 = [], img2 = []):
    '''
    By default openCV uses RANSAC with an 8 point sample.
    F is the fundamental matrix.
    Mask is a column vector with as many rows as points in imgi_points, if mask[j] == 0,
    then img1_points[j] and img2_points[j] are no longer considered as true correspondences.
    '''
    F, mask = cv.findFundamentalMat(img1_points, img2_points)
    # Getting rid of the outliers.
    img1_points = img1_points[mask.flatten() == 1]
    img2_points = img2_points[mask.flatten() == 1]

    # Computing epipolar lines in order to compute each epipole.
    lines_img1 = cv.computeCorrespondEpilines(img2_points.reshape(-1, 1, 2), 2, F)
    lines_img2 = cv.computeCorrespondEpilines(img1_points.reshape(-1, 1, 2), 1, F)
    # Computing the epipole in the first image.
    epipole1 = np.cross(lines_img1[0], lines_img1[1]).flatten()
    if img1 != []:
        draw_epilines(img1, lines_img1, img1_points, name = 'epilines1.png')
    # Compting the epipole in the second image.
    epipole2 = np.cross(lines_img2[0], lines_img2[1]).flatten()
    if img2 != []:
        draw_epilines(img2, lines_img2, img2_points, name = 'epilines2.png')
    return F, img1_points.T, img2_points.T, epipole1, epipole2

'''
Based on:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html#epipolar-geometry
'''
def draw_epilines(img, lines, points, epipole = [], name = 'epilines.png'):
    # Changing color scale.
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    columns = img.shape[1]
    for line, point in zip(lines, points):
        # a * x + b * y + c = 0
        line = line.flatten()
        # Point in the line with 0 x-coordinate.
        x1, y1 = [0, int(-line[2] / line[1])]
        # Point in the line with the maximum x-coordinate.
        x2, y2 = [columns, int(-(line[2] + line[0] * columns) / line[1])]
        img = cv.line(img, (x1, y1), (x2, y2), (0,255,0), 1)
    if epipole != []:
        img = cv.circle(img, tuple(epipole), 5, (0,255,0), -1)
    # Saving result
    plt.imsave(name, img)
    return img


# Computes the camera matrixes from the fundamental matrix (up to projective ambiguity).
def compute_cameras(F, e2, h = [], v = [], l = []):
    # Computing some ramdom projective ambiguity parameters.
    if h == []:
        h = np.random.rand(4,4)
    if v == []:
        v = np.random.rand(3,1)
    if l == []:
        l = np.random.rand(1)
        l = l

    # Computing the epipole skew-symmetric matrix.
    e2x = np.array([[0, -e2[2], e2[1]],[e2[2], 0, -e2[0]],[-e2[1], e2[0], 0]])

    # Computing the actual matrices.
    p1 = np.hstack((np.eye(3), np.zeros((3,1)))).dot(h)
    p2 = np.hstack((e2x.dot(F) + e2[:, np.newaxis].dot(v.T), l * e2[:, np.newaxis])).dot(h)
    return p1, p2

# Computes the triangulation of the scene with a least squares + svd algorithm.
def triangulate(p1, p2, img1_points, img2_points):
    X = []
    for k in range(img1_points.shape[1]):
        A = np.vstack((img1_points[1,k] * p1[2] - p1[1], p1[0] - img1_points[0,k] * p1[2], img2_points[1,k] * p2[2] - p2[1], p2[0] - img2_points[0,k] * p2[2]))
        u, s, vh = np.linalg.svd(A)
        vh = vh.T
        x = vh[:,-1]
        X.append(x)
    return np.array(X)

'''
Computes a projective 3D reconstruction (point cloud) of the scene captured by two given images.
'''
if __name__ == '__main__':
    img1, img2 = load_images(sys.argv[1], sys.argv[2])
    keypoints1, descriptors1, keypoints2, descriptors2 = compute_keypoints(img1, img2, int(sys.argv[3]))
    img1_points, img2_points = compute_correspondences(descriptors1, descriptors2, keypoints1, keypoints2, img1 = img1, img2 = img2)
    F, img1_points, img2_points, epipole1, epipole2 = compute_fundamental_matrix(img1_points, img2_points, img1 = img1, img2 = img2)
    p1, p2 = compute_cameras(F, epipole2)
    #x = cv.triangulatePoints(p1, p2, img1_points, img1_points) # (BROKEN)
    x = triangulate(p1, p2, img1_points, img2_points).T
    # Avoiding 0 divisions
    x = x[:, x[3] != 0]
    x = (x/x[3])[:3]
    v = pptk.viewer(x)
    v.set(point_size=0.01)
