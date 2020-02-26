#!/usr/bin/env python3


import sys
from std_msgs.msg import String
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/utils')
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/Frustum_PointNet')
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/pretrained_models')
import rospy
import numpy as np
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import sys
from path_planner.msg import TrafficLightArray, TrafficLight, TrafficSignArray, TrafficSign, ObstacleArray, Obstacle
import os
import math
import message_filters
import os
import numpy as np
import datetime

src_dir = "/home/autodrive/Desktop/catkin_ws/src/"
home_dir = src_dir + "keras-retinanet/"
model_dir = src_dir + "pretrained_models/"
#########----------------------------------------------

class DataLoader():

    def __init__(self):
        #self.calib = calibread(home_dir + "scripts/006330" + ".txt")
        self.calib = 0        
        self.obstacle_publisher = rospy.Publisher('/obstacle', ObstacleArray, queue_size = 1) #pathplanning
        self.traff_light_pub = rospy.Publisher('/traffic_light',TrafficLightArray, queue_size = 1)
        self.traff_sign_pub = rospy.Publisher('/traffic_sign',TrafficSignArray, queue_size = 1)
        #self.file = open(self.calib)        
        self.file = open("/home/rangeesh/catkin_ws/src/keras-retinanet/scripts/ocalib_may30" + ".txt")        
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

        self.point_cloud = None
        self.bbox = None
        self.label_string = ""
        self.frustum_point_clouds = None
        self.frustum_R = None
        self.frustum_angle = None
        self.preds = None
        self.header = None
        self.oldheader = None
        self.lock = False
        self.light_to_num = {'greenLight' : 5, 'redLight': 1,'greenLightLeft': 4,'yellowLight' : 3,'yellowLightLeft' : 3}
        self.sign_to_num = {'stopsign' : 1,  'parking_sign' : 2, 'parking_sign_handicap': 3 , 'speed_limit_10': 4, 'speed_limit_15' : 4, 'speed_limit_20' : 4, 'speed_limit_25' : 4, 'speed_limit_5' : 4, 'left_turn' : 5, 'right_turn' :6, 'do_not_enter' : 7}
        self.obs_to_num = {'bicyclist' : 1, 'car' : 3, 'person' : 4}

    def set_point_cloud(self, msg):
        x = []
        for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            x.append([point[0], point[1], point[2], 0])
        self.point_cloud = np.array(x).astype(float)
        np.save("pc.npy",self.point_cloud)

    def printobstacle(self,msg):
        self.label_string = msg.data
        pass
        print("Label: " + msg.data)
    def printstamp(self,msg):
        if self.header:
            self.oldheader = self.header
        self.header = msg
    def set_bbox(self, msg):
        #if np.max(np.array(list(msg.data))) < 1300:
        self.bbox = np.array(list(msg.data))

        ratio = float(2048./1.)
        if self.bbox.any() > 0:
            for index, point in enumerate(self.bbox):
                self.bbox[index] = point
            self.get_frustum()
        #else:
            #print("locked")



    def removeoutliers(self, bbcloud, k):
        mu = np.mean(bbcloud, axis = 0)
        sigma = np.std(bbcloud, axis = 0)
        return bbcloud[np.all(np.abs((bbcloud - mu)) < k * sigma, axis = 1)]

    
    

    def get_frustum(self):

        start = datetime.datetime.now()
        bbox = self.bbox
        point_cloud = self.point_cloud
        obstacle_array = ObstacleArray()
        traffic_light_array = TrafficLightArray()
        traffic_sign_array = TrafficSignArray()
        
        #calib = self.calib
        p2 = self.cam
        Tr_velo_to_cam_orig = self.velo_calib
        #R0_rect_orig = calib["R0_rect"]
        R0_rect = np.eye(4)
        R0_rect_orig = np.eye(3)
        R0_rect[0:3, 0:3] = R0_rect_orig
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        #print (self.label_string)
        point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
        # remove points that are located too far away from the camera:

        #40 as opposed to 80 from kitti
        point_cloud = point_cloud[point_cloud[:, 0] < 50, :]
        point_cloud = point_cloud[point_cloud[:, 2] > -1.8 , :]
        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(p2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        u_min = self.bbox[0] # (left)
        u_max = self.bbox[2] # (right)
        v_min = self.bbox[1] # (top)
        v_max = self.bbox[3] # (bottom)
        # expand the 2D bbox slightly:
        u_min_expanded = u_min
        u_max_expanded = u_max
        v_min_expanded = v_min
        v_max_expanded = v_max

        row_mask = np.logical_and(
	            np.logical_and(img_points[:, 0] >= u_min_expanded,
			           img_points[:, 0] <= u_max_expanded),
	            np.logical_and(img_points[:, 1] >= v_min_expanded,
			           img_points[:, 1] <= v_max_expanded))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :]

        #ind = np.argmin((frustum_point_cloud_xyz[:,2]))
        #minpoint = frustum_point_cloud_xyz[ind,:]
        #print (minpoint,"minpoint")

        mean = np.mean(self.removeoutliers(frustum_point_cloud_xyz,1), axis = 0)
        distance = math.sqrt(mean[0]**2 + mean[1]**2 + mean[2]**2)
        #objectlist.data = [float(self.label_string), *mean, distance]
        print (frustum_point_cloud_xyz.shape)
        np.save("original.py",frustum_point_cloud_xyz)
        r = (self.removeoutliers(frustum_point_cloud_xyz,1))
        np.save("removed.py",r)        
        obs = Obstacle()
        name = self.label_string
        print('Label: ',name)
        x = mean[2]
        y = -mean[0]
        z = mean[1]
        
        if name in self.obs_to_num:
          #obstacle_array.obstacles.append(obs)
          #self.obstacle_publisher.publish(obstacle_array)
          obs.x = x 
          obs.y = y
          obs.z = z 
          obs.type = self.obs_to_num[name]
          obstacle_array.obstacles.append(obs)
          self.obstacle_publisher.publish(obstacle_array)
        if name in self.sign_to_num:
          traff_sign_msg = TrafficSign() 
          traff_sign_msg.type = self.sign_to_num[name] 
       	if 'speed_limit' in name:
          traff_sign_msg.speed_limit = int(name.split('_')[-1])
          traff_sign_msg.x = x
          traff_sign_msg.y = y
          traffic_sign_array.signs.append(traff_sign_msg)
          self.traff_sign_pub.publish(traffic_sign_array)
        if name in self.light_to_num: #is a traffic light

          traff_light_msg = TrafficLight()
          traff_light_msg.type = self.light_to_num[name]
          traff_light_msg.x = x
          traff_light_msg.y = y
          traffic_light_array.lights.append(traff_light_msg)
          self.traff_light_pub.publish(traffic_light_array)

        print (name + " x: " + str(mean[0]) + " y: " + str(mean[1]) + " z: " + str(mean[2]))

        print ("Distance: " + str(distance))
        end = datetime.datetime.now()
      #  self.camheader = self.oldheader
        #print("time: " + str(end-start))


if __name__ == "__main__":

    data = DataLoader()

    # setup signal interrupt handler
    #signal.signal(signal.SIGINT, signalInterruptHandler)

    # variables for network
    #bridge = CvBridge()
    # setup ros semantics
    rospy.Subscriber('/campcl', PointCloud2, data.set_point_cloud)
    rospy.Subscriber('labels', String, data.printobstacle)
    rospy.Subscriber('stamp', Header, data.printstamp)

    rospy.Subscriber('/bbox', Float32MultiArray, data.set_bbox)
    rospy.init_node('lidar_node', anonymous=True)
    rospy.spin()
