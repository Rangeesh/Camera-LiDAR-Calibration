#!/usr/bin/env python3
import cv2
import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import PIL
from PIL import ImageFile
from PIL import Image as PILImage
from std_msgs.msg import Float32MultiArray
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sensor_msgs.point_cloud2
# neural network
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# ros
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

# Camera Detection ROS Node Class
class Viewcalibration:
    # Initialize class
    def __init__(self):
        #calibration file
        self.file = open("ocalib_may17" + ".txt")
        self.image = None   # image variable
        self.active = True  # should we be processing images
        self.point_cloud = None
        for line in self.file:
            data = line.split()
            if data[0] == "Tr_velo_to_cam:":
                self.velo_calib = np.array([data[1:]], dtype='float32').reshape(3,4)
            elif data[0] == "camera:":
                self.cam = np.array([data[1:]], dtype='float32').reshape(3,4)
            elif data[0] == "R0_rect:":
                self.R0_rect2 = np.array([data[1:]], dtype='float32').reshape(3,3)
        self.Tr_velo_to_cam = np.eye(4)
        self.Tr_velo_to_cam[0:3, :] = self.velo_calib
        self.R0_rect = np.eye(4) #identity matrix
        
        # CTRL+C signal handler
        signal.signal(signal.SIGINT, self.signalInterruptHandler)

        # initialize ros node
        rospy.init_node('camera_detection_node', anonymous=True)

        # set camera image topic subscriber
        self.sub_image = rospy.Subscriber('/front_camera/image_raw', Image, self.updateImage, queue_size=1)

        # ros to cv mat converter
        self.bridge = CvBridge()

        # set publishers
        self.detect_pub = rospy.Publisher('/detected', Image, queue_size = 1)
        self.bbox_pub = rospy.Publisher('/bbox', Float32MultiArray, queue_size = 1)
    def set_point_cloud(self, msg):
        x = []
        for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            x.append([point[0], point[1], point[2], 0])

        self.point_cloud = np.array(x).astype(float)

    def signalInterruptHandler(self, signum, frame):
        #print("Camera Detection - Exiting ROS Node...")

        # shut down ROS node and stop processess
        rospy.signal_shutdown("CTRL+C Signal Caught.")
        self.active = False

        sys.exit()

        try:
            self.detect_pub.publish(self.bridge.cv2_to_imgmsg(draw, "bgr8"))
        except CvBridgeError as e:
            pass#print(e)
    def updateImage(self, image_msg):
        # convert and resize
        image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        #image, scale = resize_image(image)
        # save image in class
        self.image = image
        self.get_frustum()

        
        
    def get_frustum(self):
      #image size
      image_w = self.image.shape[0]
      image_h = self.image.shape[1]
      
      point_cloud = self.point_cloud
      point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
      # remove points that are located too far away from the camera:
        
      #40 as opposed to 80 from kitti
      point_cloud = point_cloud[point_cloud[:, 0] < 15, :]
      point_cloud_xyz = point_cloud[:, 0:3]
      point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
      point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

      # project the points onto the image plane (homogeneous coords):
      img_points_hom = np.dot(self.cam, np.dot(self.R0_rect, np.dot(self.Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))

      # normalize:
      img_points = np.zeros((img_points_hom.shape[0], 2))
      img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
      img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]
      
     # print img_points_hom
     # print point_cloud_xyz_hom
      
      row_mask = np.logical_and(
          np.logical_and(img_points[:, 0] >= 0,
                         img_points[:, 0] <= image_w),
          np.logical_and(img_points[:, 1] >= 0,
img_points[:, 1] <= image_h))
      #trim points that may have been overloaded/not filtered
      #test = img_points_hom[row_mask,:]
      #point_cloud_xyz = point_cloud_xyz[row_mask,:]
      img_points = img_points[row_mask, :]
      #print test, point_cloud_xyz
      #create black image to match coordinates of image
      pclimage = np.zeros((image_h,image_w,3), np.uint8)
      
    
      #img_points = 2D array
      
      #needed for cv2 array coloring
      
      img_points = img_points.astype(int)
      for i in range(1,len(img_points)):
        #color 
        self.image[img_points[i][1],img_points[i][0],0] = 0
        self.image[img_points[i][1],img_points[i][0],1] = 255
        self.image[img_points[i][1],img_points[i][0],2] = 0
        pclimage[img_points[i][1],img_points[i][0],0] = 0
        pclimage[img_points[i][1],img_points[i][0],1] = 255
        pclimage[img_points[i][1],img_points[i][0],2] = 0
        
      #show point cloud
      cv2.namedWindow("point cloud", cv2.WINDOW_NORMAL) 
      cv2.resizeWindow("point cloud", 1000,1000)
      cv2.imshow("point cloud",pclimage)

      cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
      cv2.resizeWindow("image", 1000,1000)
      cv2.imshow("image",self.image)
      cv2.waitKey(0)
      #cv2.destroyAllWindows()

if __name__ == "__main__":
    # start camera detection ros node
    print('me')
    v = Viewcalibration()
    rospy.Subscriber('/lidar_center/velodyne_points', PointCloud2, v.set_point_cloud)
    rospy.spin()

