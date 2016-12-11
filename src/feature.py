"""
Feature Detector/Descriptor Visualization
Licence: BSD
Author : Hoang Anh Nguyen
"""

import cv2
import numpy as np
import argparse, sys

#--------------------------------------------------------
#--------------------------------------------------------
# Class provide an interface to facilitate the use of Feature 
# detector/descriptors with OpenCV.
class Feature:
    
    def __init__(self, image_name, detectorId, descriptorId):
        # read image file and convert it to grayscale
        self.origin_image = cv2.imread(image_name)
        self.gray_image   = cv2.cvtColor(self.origin_image,
                                         cv2.COLOR_BGR2GRAY)
        
        # setup combination detector - descriptor
        print(Detector.getDetectorNameBasedOnId(detectorId))
        print(Descriptor.getDescriptorNameBasedOnId(descriptorId))
        self.detector    = eval(Detector.getDetectorNameBasedOnId(detectorId))()
        self.descriptor  = eval(Descriptor.getDescriptorNameBasedOnId(descriptorId))()
        self.flann       = None
        
    #--------------------------------------------------------
    # Main methods
    #--------------------------------------------------------
    
    # Extract features in the image using a specific Detector/Descriptor
    # input:    showResult
    # return:   number of keypoints
    #           description of each keypoint
    def extract(self, showResult = False):
        if self.detector:
            # detect keypoints and extract descriptor of each keypoint
            self.detector.detect(self.gray_image)
            self.descriptor.describe(self.gray_image,
                                     self.detector.keypoints)
            print("keypoints: {}, descriptors: {}"\
                  .format(len(self.detector.keypoints),
                          self.descriptor.descriptions.size))
            if showResult:
                return self.showFeatures()
            
        return None
    
    # Match current feature with another
    # input:    target Feature
    # return:   matching mask between original vs. target features
    def match(self, feature2, showResult = False):
        if not self.flann:
            self.initializeFlannMatcher()
        
        # FLANN parameters
        self.matches = self.flann.knnMatch(self.descriptor.descriptions,
                                           feature2.descriptor.descriptions,
                                           k=2)

        # Need to draw only good matches, so create a mask
        self.matchesMask = [[0,0] for i in xrange(len(self.matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(self.matches):
            if m.distance < 0.7 * n.distance:
                self.matchesMask[i]=[1, 0]
        
        if showResult:
            return self.showMatches(feature2)
        
        return None
    
    #--------------------------------------------------------
    # Inner methods
    #--------------------------------------------------------
    
    # Using Flann for descriptor matching
    def initializeFlannMatcher(self):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6,
                            key_size = 12,
                            multi_probe_level = 1)
        # need to check type of key
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # draw keypoints & descriptors on origin image
    def showFeatures(self):
        if not self.detector.keypoints:
            print('Keypoint not found!')
            return None
        
        cv2.drawKeypoints(self.origin_image,
                          self.detector.keypoints,
                          self.origin_image,
                          (0, 255, 0))
        return self.origin_image
    
    # show keypoint matching betwen 2 images
    def showMatches(self, feature2):
        if not self.detector.keypoints or not feature2.detector.keypoints:
            print('Keypoint not found!')
            return
        
        if not self.matchesMask or not self.matches:
            print('Invalid matches!')
            return
        
        draw_params = dict(matchColor = (0, 255, 0),
                           singlePointColor = (255, 0, 0),
                           matchesMask = self.matchesMask,
                           flags = 0)

        output = cv2.drawMatchesKnn(self.origin_image,
                                    self.detector.keypoints,
                                    feature2.origin_image,
                                    feature2.detector.keypoints,
                                    self.matches,
                                    None,
                                    **draw_params)
        return output
    
#-------------------------------------------------------
#----------------- Keypoint Detector -------------------
class Detector:
    
    def __init__(self):
        self.detector  = None
        self.keypoints = None
    
    def detect(self, gray_image):
        if self.detector:
            self.keypoints = self.detector.detect(gray_image, None)
            
    # Static methods
    @staticmethod
    def getDetectorNameBasedOnId(index):
        if index < 1 or index > 9:
            print('Invalid Detector!')
            return None
        
        return [
            'AkazeDetector',
            'KazeDetector',
            'FASTDetector',
            'BRISKDetector',
            'ORBDetector',
            'MSDDetector',
            'StarDetector',
            'AGASTDetector',
            'GFTTDetector'
        ][index - 1]
        
# Akaze (http://docs.opencv.org/trunk/d8/d30/classcv_1_1AKAZE.html)
class AkazeDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.AKAZE_create()
        
# Kaze (http://docs.opencv.org/trunk/d3/d61/classcv_1_1KAZE.html)
class KazeDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.KAZE_create()
        
# FAST (http://docs.opencv.org/trunk/df/d74/classcv_1_1FastFeatureDetector.html)
class FASTDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.FastFeatureDetector_create()
        
# BRISK (http://docs.opencv.org/trunk/de/dbf/classcv_1_1BRISK.html)
class BRISKDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.BRISK_create()

# ORB (http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html)
class ORBDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.ORB_create()
    
# MSD (Maximal Self-Dissimilarity) 
class MSDDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.xfeatures2d.MSDDetector_create()

# StarDetector
class StarDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.xfeatures2d.StarDetector_create()

# AGAST (http://docs.opencv.org/trunk/d7/d19/classcv_1_1AgastFeatureDetector.html)
class AGASTDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.AgastFeatureDetector_create()

# GFTT (http://docs.opencv.org/trunk/df/d21/classcv_1_1GFTTDetector.html)
class GFTTDetector(Detector):
    
    def __init__(self):
        Detector.__init__(self)
        self.detector = cv2.GFTTDetector_create()

#--------------------------------------------------------
#------------------- Descriptor -------------------------
class Descriptor:
    
    def __init__(self):
        self.descriptor   = None
        self.descriptions = None
    
    def describe(self, gray_image, keypoints):
        if self.descriptor:
            [__, self.descriptions] = self.descriptor.compute(gray_image, keypoints)
            
    @staticmethod
    def getDescriptorNameBasedOnId(index):
        if index < 1 or index > 11:
            print('Invalid Descriptor')
            return None
        
        return [
            'AKAZEDescriptor',
            'KAZEDescriptor',
            'BRISKDescriptor',
            'ORBDescriptor',
            'BRIEFDescriptor',
            'DAISYDescriptor',
            'BoostDescriptor',
            'FREAKDescriptor',
            'LATCHDescriptor',
            'LUCIDDescriptor',
            'VGGDescriptor'
        ][index - 1]
            
# AKAZE (http://docs.opencv.org/trunk/d8/d30/classcv_1_1AKAZE.html)
class AKAZEDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.AKAZE_create()
        
# KAZE (http://docs.opencv.org/trunk/d3/d61/classcv_1_1KAZE.html)
class KAZEDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.KAZE_create()
        
# BRISK (http://docs.opencv.org/trunk/de/dbf/classcv_1_1BRISK.html)
class BRISKDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.BRISK_create()
        
# ORB (http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html)
class ORBDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.ORB_create()
        
# BRIEF (http://docs.opencv.org/trunk/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html)
class BRIEFDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        
# DAISY (http://docs.opencv.org/trunk/d9/d37/classcv_1_1xfeatures2d_1_1DAISY.html)
class DAISYDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.xfeatures2d.DAISY_create()

#BoostDesc (http://docs.opencv.org/trunk/d1/dfd/classcv_1_1xfeatures2d_1_1BoostDesc.html)
class BoostDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.xfeatures2d.BoostDesc_create()
        
# FREAK (http://docs.opencv.org/trunk/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html)
class FREAKDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.xfeatures2d.FREAK_create()
        
# LATCH (http://docs.opencv.org/trunk/d6/d36/classcv_1_1xfeatures2d_1_1LATCH.html)
class LATCHDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.xfeatures2d.LATCH_create()

# LUCID (http://docs.opencv.org/trunk/d4/d86/classcv_1_1xfeatures2d_1_1LUCID.html)
class LUCIDDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.xfeatures2d.LUCID_create()
        
# VGG (http://docs.opencv.org/trunk/d6/d00/classcv_1_1xfeatures2d_1_1VGG.html)
class VGGDescriptor(Descriptor):
    
    def __init__(self):
        Descriptor.__init__(self)
        self.descriptor = cv2.xfeatures2d.VGG_create()
        
#--------------------------------------------------------
#------------------- Blob Detector-----------------------
# SimpleBlobDetector

# MSER: Region detector, not key-points
class MSERFeature(Feature):
    
    def __init__(self, image_name):
        Feature.__init__(self, image_name)
        
    def detect(self, showResult = False):
        self.detector = cv2.MSER_create()

        self.blobs = self.detector.detectRegions(
            self.gray_image,
            None)
        if showResult:
            return self.showFeatures()
    
    def showFeatures(self):
        img = self.origin_image.copy()
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in self.blobs]
        cv2.polylines(img, hulls, 1, (0, 255, 0))
        return img
    
    def match(self, feature2, showResult = False):
        print("MSER can't do keypoints matching")
        return None
        
#--------------------------------------------------------
#--------------------------------------------------------
def main(argv):
    # Define argument list. Example:
    # python feature.py -t 1
    #                   -k 1
    #                   -d 1
    #                   -i1 resources/template1.jpg
    #                   -i2 resources/test_template1.jpg
    #                   -o .
    parser = argparse.ArgumentParser(description='Feature 2D')
    parser.add_argument('-t','--task',
                        help="""Specify task
                        1: Detecting
                        2: Matching
                        """,
                        required=True)
    parser.add_argument('-k', '--keypoint',
                        help="""Specify keypoint detectors
                        1: AKAZE,
                        2: KAZE,
                        3: FAST,
                        4: BRISK,
                        5: ORB,
                        6: MSD,     x
                        7: Star,
                        8: AGAST,
                        9: GFTT
                        """,
                        required=True)
    parser.add_argument('-d', '--descriptor',
                        help="""Specify keypoint detectors
                        1: AKAZE,
                        2: KAZE,
                        3: BRISK,
                        4: ORB,
                        5: BRIEF,   x
                        6: DAISY,   x
                        7: Boost,
                        8: FREAK,
                        9: LATCH,
                        10: LUCID,
                        11: VGG
                        """,
                        required=True)
    parser.add_argument('-i1','--input1',
                        help='Input image 1',
                        required=True)
    parser.add_argument('-i2','--input2',
                        help='Input image 2 (for matching)')
    parser.add_argument('-o','--output',
                        help='Ouput location',
                        required=True)
    args = vars(parser.parse_args())
    
    # extract arguments
    task         = int(args['task'])
    detectorId   = int(args['keypoint'])
    descriptorId = int(args['descriptor'])
    print('Argument parsed: ', task, detectorId, descriptorId)
    
    if task != 1 and task != 2:
        print("Invalid task: " + args['task'])
        return
    
    # find keypoints & build descriptor on input1 image
    feature1 = Feature(args['input1'],
                       detectorId,
                       descriptorId)
    output = feature1.extract(task == 1)
    
    if task == 2:        
        if not args['input2']:
            print("Missing second input image for matching!")
            return
            
        # find keypoints & build descriptor on input2 image
        feature2 = Feature(args['input2'],
                           detectorId,
                           descriptorId)
        feature2.extract()

        # matching feature bwt 2 images, and save result
        output = feature1.match(feature2, True)
    
    # save output
    cv2.imwrite(args['output'] + '/output.jpg', output)
    print("Output saved!")
    
if __name__ == '__main__':
    main(sys.argv)
