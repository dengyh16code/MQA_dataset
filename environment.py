
try:
    from vrep import*
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import os
import sys
import numpy.random as random
import numpy as np
import math
from collections import defaultdict
import PIL.Image as Image
import array
import json
import shapely
from shapely.geometry import Polygon
import cv2 as cv

class Camera(object):
    """
        # kinect camera in simulation
    """
    def __init__(self, clientID):
        """
            Initialize the Camera in simulation
        """
        self.RAD2EDG = 180 / math.pi
        self.EDG2RAD = math.pi / 180
        self.Save_IMG = True
        self.Save_PATH_COLOR = r'./color'
        self.Save_PATH_DEPTH = r'./depth'
        self.Dis_FAR = 10
        self.depth_scale = 1000
        self.Img_WIDTH = 224
        self.Img_HEIGHT = 224
        self.border_pos = [120,375,100,430]# [68,324,112,388] #up down left right of the box
        self.theta = 70
        self.Camera_NAME = r'kinect'
        self.Camera_RGB_NAME = r'kinect_rgb'
        self.Camera_DEPTH_NAME = r'kinect_depth'
        self.clientID = clientID
        self._setup_sim_camera()
        self._mkdir_save(self.Save_PATH_COLOR)
        self._mkdir_save(self.Save_PATH_DEPTH)

    def _mkdir_save(self, path_name):
        if not os.path.isdir(path_name):         
            os.mkdir(path_name)

    def _euler2rotm(self,theta):
        """
            -- Get rotation matrix from euler angles
        """
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])         
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])            
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R


    def _setup_sim_camera(self):
        """
            -- Get some param and handles from the simulation scene
            and set necessary parameter for camera
        """
        # Get handle to camera
        _, self.cam_handle = simxGetObjectHandle(self.clientID, self.Camera_NAME, simx_opmode_oneshot_wait)
        _, self.kinectRGB_handle = simxGetObjectHandle(self.clientID, self.Camera_RGB_NAME, simx_opmode_oneshot_wait)
        _, self.kinectDepth_handle = simxGetObjectHandle(self.clientID, self.Camera_DEPTH_NAME, simx_opmode_oneshot_wait)
        # Get camera pose and intrinsics in simulation
        _, self.cam_position = simxGetObjectPosition(self.clientID, self.cam_handle, -1, simx_opmode_oneshot_wait)
        _, cam_orientation = simxGetObjectOrientation(self.clientID, self.cam_handle, -1, simx_opmode_oneshot_wait)

        self.cam_trans = np.eye(4,4)
        self.cam_trans[0:3,3] = np.asarray(self.cam_position)
        self.cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        self.cam_rotm = np.eye(4,4)
        self.cam_rotm[0:3,0:3] = np.linalg.inv(self._euler2rotm(cam_orientation))
        self.cam_pose = np.dot(self.cam_trans, self.cam_rotm) # Compute rigid transformation representating camera pose
        self._intri_camera()

    def _intri_camera(self):  #the paramter of camera
        """
            Calculate the intrinstic parameters of camera
        """
        fx = -self.Img_WIDTH/(2.0 * math.tan(self.theta * self.EDG2RAD / 2.0))
        fy = fx
        u0 = self.Img_HEIGHT/ 2
        v0 = self.Img_WIDTH / 2
        self.intri = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0, 0, 1]])


    def get_camera_data(self):
        """
            -- Read images data from vrep and convert into np array
        """
        # Get color image from simulation
        res, resolution, raw_image = simxGetVisionSensorImage(self.clientID, self.kinectRGB_handle, 0, simx_opmode_oneshot_wait)
        # self._error_catch(res)
        color_img = np.array(raw_image, dtype=np.uint8)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.flipud(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        res, resolution, depth_buffer = simxGetVisionSensorDepthBuffer(self.clientID, self.kinectDepth_handle, simx_opmode_oneshot_wait)
        # self._error_catch(res)
        depth_img = np.array(depth_buffer)
        #print(depth_img)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.flipud(depth_img)
        depth_img[depth_img < 0] = 0
        depth_img[depth_img > 1] = 0.9999
        depth_img = depth_img * self.Dis_FAR * self.depth_scale
        self.cur_depth = depth_img
        return depth_img, color_img

    def save_image(self, cur_depth, cur_color, img_idx):
        """
            -- Save Color&Depth images
        """
        img = Image.fromarray(cur_color.astype('uint8')).convert('RGB')
        img_path = os.path.join(self.Save_PATH_COLOR, str(img_idx) + '_Rgb.png')
        img.save(img_path)
        depth_img = Image.fromarray(cur_depth.astype(np.uint32),mode='I')
        depth_path = os.path.join(self.Save_PATH_DEPTH, str(img_idx) + '_Depth.png')
        depth_img.save(depth_path)

        return depth_path, img_path

    def _error_catch(self, res):
        """
            -- Deal with error unexcepted
        """
        if res == simx_return_ok:
            print ("--- Image Exist!!!")
        elif res == simx_return_novalue_flag:
            print ("--- No image yet")
        else:
            print ("--- Error Raise")


    def pixel2ur5(self, u, v, ur5_position, push_depth, depth = 0.0, is_dst = True):
        """
            from pixel u,v and correspondent depth z -> coor in ur5 coordinate (x,y,z)
        """
        if is_dst == False:
            depth = self.cur_depth[int(u)][int(v)] / self.depth_scale

        x = depth * (u - self.intri[0][2]) / self.intri[0][0]
        y = depth * (v - self.intri[1][2]) / self.intri[1][1]
        camera_coor = np.array([x, y, depth - push_depth])
        """
            from camera coor to ur5 coor
            Notice the camera faces the plain directly and we needn't convert the depth to real z
        """
        camera_coor[2] = - camera_coor[2]
        location = camera_coor + self.cam_position - np.asarray(ur5_position)
        return location, depth

    def world2pixel(self,location):
        """
            from  coor in world coordinate (x,y,z) to pixel u.v
        """
        x=location[0]
        y=location[1]
        z=0.75
        # extrinsic parameter
        if x<0.0 or x> 1.0 or abs(y)>0.7 or z<0.7:
            return [1024,1024]
           
        #print(self.cam_position)

        z_1 = self.cam_position[2]-z
        x_1 = x-self.cam_position[0]
        y_1 = y-self.cam_position[1]

        # internal parameter
        u = int((x_1 / z_1)*self.intri[0][0] +self.intri[0][2])    #u    x_1
        v = int((y_1 / z_1)*self.intri[1][1] +self.intri[1][2])    #v    y_1

        if u<0 or v<0:
            return [1024,1024]

        return [u,v]
        

    '''
        x=location[0]
        y=location[1]
        z=0.75
        # extrinsic parameter
        if x<0.0 or x> 1.0 or abs(y)>0.7 or z<0.7:
            return [1024,1024]
           

        z_1 = self.cam_position[2]-z
        x_1 = x-self.cam_position[0]
        y_1 = y-self.cam_position[1]

        f= -self.Img_WIDTH/(2.0 * math.tan(self.theta * self.EDG2RAD / 2.0))

        u = f/z_1 * y_1  + self.intri[0][2]

        v = f/z_1 * x_1 + self.intri[1][2]
    '''


        




    def pixel2world(self, u, v,  push_depth = 0):
        """
            from pixel u,v and correspondent depth z -> coor in world coordinate (x,y,z)
            
        """
        if u >0:
            u = int(u)%224
        else:
            u = 0

        if v>0:
            v = int(v)%224
        else:
            v= 0

        depth = self.cur_depth[int(u)][int(v)] / self.depth_scale
        x = depth * (u - self.intri[0][2]) / self.intri[0][0]
        y = depth * (v - self.intri[1][2]) / self.intri[1][1]
        camera_coor = np.array([x, y, depth-push_depth])
        """
            from camera coor to world coor
            Notice the camera faces the plain directly and we needn't convert the depth to real z
        """
        camera_coor[2] = - camera_coor[2]
        location = camera_coor + self.cam_position
        return location

class UR5(object):
    def __init__(self,testing_file='table-00-scene-00.txt',obj_num=20):
        #test
        self.testing_file = testing_file
        self.targetPosition = np.zeros(3,dtype = np.float)
        self.targetQuaternion = np.array([0.707,0,0.707,0])
        self.baseName = r'UR5'
        self.IkName = r'UR5_ikTip'
        table_file = 'data/box.txt'
        bound_dir = "data/boundary_size.json"
        bound_file = open(bound_dir,encoding='utf-8')
        self.bound_dic = json.load(bound_file)
        file = open(table_file, 'r')
        file_content = file.readlines()
        file.close()
        self.table_para = file_content[0].split()    
        self.workspace_limits = np.asarray([[float(self.table_para[0]), float(self.table_para[1])], [float(self.table_para[2]), float(self.table_para[3])] ])
        self.drop_height =0.1
        self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                                       [89.0, 161.0, 79.0],  # green
                                       [156, 117, 95],  # brown
                                       [242, 142, 43],  # orange
                                       [237.0, 201.0, 72.0],  # yellow
                                       [186, 176, 172],  # gray
                                       [255.0, 87.0, 89.0],  # red
                                       [176, 122, 161],  # purple
                                       [118, 183, 178],  # cyan
                                       [255, 157, 167]]) / 255.0  # pink
        # Read files in object mesh directory
        self.test_file_dir = os.path.abspath('data/test_cases/')
        self.test_preset_file = os.path.join(self.test_file_dir, self.testing_file)
        self.obj_mesh_dir= os.path.abspath('data/mesh/')
        self.obj_num = obj_num
        self.obj_dict = defaultdict(dict)

        simxFinish(-1)  # just in case, close all opened connections
        self.clientID = simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP
        if self.clientID != -1:
            print ('Connected to remote API server')
            # If testing, read object meshes and poses from test case file
            scene_file = open(self.test_preset_file, 'r')
            file_content = scene_file.readlines()
            self.test_obj_mesh_files = []
            self.test_obj_name = []
            self.test_obj_type = []
            self.test_obj_mesh_colors = []
            self.test_obj_positions = []
            self.test_obj_orientations = []
            for i in range(self.obj_num):
                file_content_curr_object = file_content[i].split()
                self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir, file_content_curr_object[0]))
                self.test_obj_name.append(file_content_curr_object[0])
                self.test_obj_type.append(file_content_curr_object[0][:-5])
                self.test_obj_positions.append(
                    [float(file_content_curr_object[1]), float(file_content_curr_object[2]),
                        float(file_content_curr_object[3])])
                self.test_obj_orientations.append(
                    [float(file_content_curr_object[4]), float(file_content_curr_object[5]),
                        float(file_content_curr_object[6])])
            scene_file.close()
            simxStartSimulation(self.clientID, simx_opmode_blocking)
            #self.add_objects()
        else:
            print ('Failed connecting to remote API server')
        _, self.ur5_handle = simxGetObjectHandle(self.clientID,self.baseName,simx_opmode_oneshot_wait)
        _, self.ur5_position = simxGetObjectPosition(self.clientID,self.ur5_handle,-1,simx_opmode_oneshot_wait)
        self.Ik_handle = simxGetObjectHandle(self.clientID,self.IkName,simx_opmode_oneshot_wait)
        self.add_objects()
        self.ankleinit()


    def ankleinit(self):
        """
            # initial the ankle angle for ur5
        """
        simxSynchronousTrigger(self.clientID) 
        simxPauseCommunication(self.clientID, True)
        simxSetIntegerSignal(self.clientID, 'ICECUBE_0', 11, simx_opmode_oneshot)
        simxPauseCommunication(self.clientID, False)
        simxSynchronousTrigger(self.clientID)
        simxGetPingTime(self.clientID)
        # pause for 1s
        time.sleep(1)


    def disconnect(self):
        """
            # disconnect from v-rep
            # and stop simulation
        """
        simxStopSimulation(self.clientID,simx_opmode_oneshot)
        time.sleep(2)
        simxFinish(self.clientID)
        print ('Simulation ended!')

    def get_clientID(self):
        return self.clientID
        

    def ur5push(self, move_begin, move_to):
        """
            The action of the ur5 in a single push action including:
            Get to push beginning
            Push to the destination
            Return to the init pose
        """

        time.sleep(1)       
        self.ur5moveto(move_begin)
        time.sleep(0.5)
        self.ur5moveto(move_to)
        time.sleep(0.5)

        # Return to the initial pose
        self.ankleinit()
        time.sleep(0.5)


    def ur5moveto(self, dst_location):
        """
            Push the ur5 hand to the location of dst_location
        """
        simxSynchronousTrigger(self.clientID)
        self.targetPosition = dst_location
        simxPauseCommunication(self.clientID, True)
        simxSetIntegerSignal(self.clientID, 'ICECUBE_0', 21, simx_opmode_oneshot)
        for i in range(3):
            simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+1),self.targetPosition[i],simx_opmode_oneshot)
        for i in range(4):
            simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+4),self.targetQuaternion[i], simx_opmode_oneshot)
        simxPauseCommunication(self.clientID, False)
        simxSynchronousTrigger(self.clientID)
        simxGetPingTime(self.clientID)



    def add_objects(self):
        # Add objects to robot workspace at x,y location and orientation
        self.object_handles = []
        for i in range(self.obj_num):
            curr_shape_name = 'shape'+str(i)
            curr_mesh_file = self.test_obj_mesh_files[i]
            curr_obj_type = self.test_obj_type[i]
            curr_type = 't'
            object_position = [self.test_obj_positions[i][0], self.test_obj_positions[i][1], self.test_obj_positions[i][2]]
            object_orientation = [self.test_obj_orientations[i][0]*np.pi/180, self.test_obj_orientations[i][1]*np.pi/180, self.test_obj_orientations[i][2]*np.pi/180]
            #print (object_position + object_orientation, [curr_mesh_file, curr_shape_name])
            ret_resp,ret_ints,_,ret_strings,_ = simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation, [curr_mesh_file, curr_shape_name,curr_type], bytearray(), simx_opmode_blocking)
            time.sleep(1)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                print (ret_strings)
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)


    
    def get_obj_positions_and_orientations(self):
        
        for i in range(self.obj_num):
            obj_handle = self.object_handles[i]
            self.obj_dict[i]['handle'] = obj_handle
            _, object_position = simxGetObjectPosition(self.clientID, obj_handle, -1, simx_opmode_blocking)
            _, object_orientation = simxGetObjectOrientation(self.clientID, obj_handle, -1, simx_opmode_blocking)

            self.obj_dict[i]['position'] = object_position
            self.obj_dict[i]['orientation'] = object_orientation
            object_matrix = self.euler2rotm(object_orientation, object_position)
            # object_matrix = self.euler2rotm_1(i)
            self.obj_dict[i]['matrix'] = object_matrix

            obj_name = self.test_obj_name[i]
            self.obj_dict[i]['name'] = obj_name
            self.obj_dict[i]['boundary_size'] = self.bound_dic[obj_name]

            self.obj_dict[i]['rect'],self.obj_dict[i]['boundary']= self.caculate_projection_rect(object_matrix,self.bound_dic[obj_name])
            
            #print(obj_name)
            #print(object_position)
            #print(object_orientation)
            #print(self.obj_dict[i]['boundary'])

        return self.obj_dict

    def adjust_obj_positions_and_oritentations(self):
        for i in range(self.obj_num):
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 1]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            handle = self.object_handles[i]
            simxPauseCommunication(self.clientID,True)
            simxSetObjectOrientation(self.clientID,handle,-1,object_orientation,simx_opmode_oneshot)
            simxPauseCommunication(self.clientID,False)
            simxPauseCommunication(self.clientID,True)
            simxSetObjectPosition(self.clientID,handle,-1,object_position,simx_opmode_oneshot)
            simxPauseCommunication(self.clientID,False)
            time.sleep(1)



    def euler2rotm(self,theta,position):
        """
            -- Get rotation matrix from euler angles
        """
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])         
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])            
        R = np.dot(R_z, np.dot( R_y, R_x ))
        position_get = np.array([position])
        R1 = np.vstack((R,position_get))
     
        return R1


    def caculate_projection_rect(self,object_matrix,boundary_size):
        obj_points =np.array( [
                      [boundary_size[0]/2,boundary_size[0]/2,-boundary_size[0]/2,-boundary_size[0]/2,
                      boundary_size[0]/2,boundary_size[0]/2,-boundary_size[0]/2,-boundary_size[0]/2],

                      [boundary_size[1]/2,-boundary_size[0]/2,boundary_size[1]/2,-boundary_size[0]/2,
                      boundary_size[1]/2,-boundary_size[0]/2,boundary_size[1]/2,-boundary_size[0]/2,],

                      [boundary_size[2]/2,boundary_size[2]/2,boundary_size[2]/2,boundary_size[2]/2,
                      -boundary_size[2]/2,-boundary_size[2]/2,-boundary_size[2]/2,-boundary_size[2]/2],

                      [1,1,1,1,1,1,1,1]
                     ])

        obj_points = obj_points.T
        obj_points_transform = np.dot(obj_points,object_matrix)
        obj_points_transform = np.array(obj_points_transform)
        obj_points_transform = obj_points_transform.reshape(8,3)
        obj_x_array = obj_points_transform[:,0].T
        obj_y_array = obj_points_transform[:,1].T



        x_max_point = np.where(obj_x_array == np.max(obj_x_array))[0][0]
        x_min_point = np.where(obj_x_array == np.min(obj_x_array))[0][0]
        y_max_point = np.where(obj_y_array == np.max(obj_y_array))[0][0]
        y_min_point = np.where(obj_y_array == np.min(obj_y_array))[0][0]



        x_max = obj_points_transform[x_max_point]
        x_min = obj_points_transform[x_min_point]
        y_max = obj_points_transform[y_max_point]
        y_min = obj_points_transform[y_min_point]


        rect = [

        obj_points_transform[x_max_point][0],obj_points_transform[x_max_point][1],   
        obj_points_transform[x_min_point][0],obj_points_transform[x_min_point][1],
        obj_points_transform[y_max_point][0],obj_points_transform[y_max_point][1],
        obj_points_transform[y_min_point][0],obj_points_transform[y_min_point][1]

        ]

        
        rect1 = np.array(rect).reshape(4,2)
        boundary = [x_min,y_max,x_max,y_min]
        poly = Polygon(rect1).convex_hull

        return poly,boundary


    def check_overlap(self,target_order,obj_dict):
        # find the bound of the obj_target
        target_rect =self.obj_dict[target_order]['rect']
        target_rect_area = target_rect.area
        overlap_rate = 0
        overlap_order = target_order

        for order in range(self.obj_num):
            # check if the ith obj we are looking at is obj_target
            if order == target_order:
                continue
            # a different obj
            else:
                obj_type =  self.obj_dict[order]['name']
                obj_type = obj_type[:-5]
                cal_rect = self.obj_dict[order]['rect']
                if not target_rect.intersection(cal_rect): # no overlap
                    continue
                else:
                    overlap_area = target_rect.intersection(cal_rect).area
                    rate_temp = overlap_area / target_rect_area
                    if rate_temp > overlap_rate:
                        overlap_rate = rate_temp
                        overlap_order = order
                        
        return overlap_rate,overlap_order
    

 



class Environment(object):
    """
         simulation environment 
    """
    def __init__(self,testing_file='group-00-scene-00.txt',obj_num =20 ):

        # initial the ur5 arm in simulation
        self.obj_num = obj_num
        self.ur5 = UR5(testing_file=testing_file,obj_num=self.obj_num)
        self.ur5.ankleinit()
        self.ur5_location = self.ur5.ur5_position
        # initial the camera in simulation
        self.clientID = self.ur5.get_clientID()
        self.camera = Camera(self.clientID)
        time.sleep(1)
        self.obj_dic = self.ur5.get_obj_positions_and_orientations()

        print('\n [*] Initialize the simulation environment')

    def new_scene(self,group_num,scene_num):     # choose a scene and related questions from the dataset

        scene_name = 'group-'+ '0'+str(group_num) + '-scene-'+'0'+ str(scene_num) + '.txt'

        ques_encode_file_name =  'group-'+ '0'+str(group_num) + '-scene-'+'0'+ str(scene_num) + '-encodeques'+'.json'
        ques_encode_file_dir = os.path.abspath('data/encode_ques/')
        ques_encode_full_file = os.path.join(ques_encode_file_dir, ques_encode_file_name)
        ques_encode_file = open(ques_encode_full_file,'r',encoding = 'utf-8')
        all_encode_ques = json.load(ques_encode_file)
        print(all_encode_ques)

        ques_file_name =  'group-'+ '0'+str(group_num) + '-scene-'+'0'+ str(scene_num) + '-ques'+'.json'
        ques_file_dir = os.path.abspath('data/ques/')
        ques_full_file = os.path.join(ques_file_dir, ques_file_name)
        ques_file = open(ques_full_file,'r',encoding = 'utf-8')
        all_ques = json.load(ques_file)
        print(all_ques)

        self.close()
        self.obj_num = int(scene_num//3) * 15 + 20
        print("obj_num",self.obj_num)
        print(scene_name)
        self.ur5 = UR5(testing_file=scene_name,obj_num=self.obj_num)
        self.ur5.ankleinit()
        self.ur5_location = self.ur5.ur5_position
        # initial the camera in simulation
        self.clientID = self.ur5.get_clientID()
        self.camera = Camera(self.clientID)
        time.sleep(1)
        self.obj_dic = self.ur5.get_obj_positions_and_orientations()

        print('\n [*] Initialize the simulation environment')

        depth_img,rgb_img =  self.camera.get_camera_data()
        return rgb_img, depth_img, all_ques,all_encode_ques




    def act(self,action_location,target_name,ques_type,reward_type,reward_weight):   #1:push 2:suck 3:loose
        loca_ori = action_location // (28*28)
        loca_x = (action_location % (28*28))//28
        loca_y = (action_location % (28*28))%28
        action = [loca_ori,8*loca_x,8*loca_y]
        overlap_list_before = []
        overlap_list_after = []
        position_list_before = []
        position_list_after = []
        target_index =  [i for i,x in enumerate(self.ur5.test_obj_type) if x == target_name]
        push_depth=-0.1
        push_dis = 224/4
        ori = action[0]*math.pi/4
        start_point = [action[1],action[2]]
        end_point = [(action[1] + push_dis * math.cos(ori)),(action[2] + push_dis * math.sin(ori))]
        move_begin = self.camera.pixel2world(start_point[0], start_point[1], push_depth)
        move_to = self.camera.pixel2world(end_point[0], end_point[1], push_depth)
        

        for one_target_index in target_index:
            overlap_rate,_ = self.ur5.check_overlap(one_target_index,self.obj_dic)
            overlap_list_before.append(overlap_rate)
        for obj_index in range(self.obj_num):
            position_list_before.append(self.ur5.obj_dict[obj_index]['position'])


        if loca_ori != 8:  # the output action is not stop
            self.ur5.ur5push(move_begin,move_to)
            time.sleep(2)
            #print('\n -- Push from {} to {}' .format(start_point,end_point))
            self.obj_dic = self.ur5.get_obj_positions_and_orientations()   #take action and update obj_dic


        for one_target_index in target_index:
            overlap_rate,_ = self.ur5.check_overlap(one_target_index,self.obj_dic)
            overlap_list_after.append(overlap_rate)
        for obj_index in range(self.obj_num):
            position_list_after.append(self.ur5.obj_dict[obj_index]['position'])


        reward_para = [overlap_list_before,overlap_list_after,position_list_before,position_list_after]

        reward,terminal,reward_e,reward_q = self.reward_cal(loca_ori,ques_type,reward_para,reward_type,reward_weight)
        depth_image_after,rgb_image_after =  self.camera.get_camera_data()

        return rgb_image_after,depth_image_after,reward,terminal,reward_e,reward_q 

    def reward_cal(self,action_type,ques_type,reward_para,reward_type,reward_weight):
        reward_e = 0
        reward_q = 0
        terminal = 0
        for i in range(self.obj_num):
            p_i = np.array(reward_para[2][i])
            p_i_1 = np.array(reward_para[3][i])
            reward_e += np.linalg.norm(p_i-p_i_1)/(np.linalg.norm(p_i)*self.obj_num)

        predict_stop_sig = int(action_type==8)
        if ques_type in ['exist_negative','exist_positive']:
            if len(reward_para[0]) == 0:
                true_stop_sig  = 1
            elif min(reward_para[0]) <0.1:
                true_stop_sig = 1
            else:
                true_stop_sig = 0
        elif ques_type in ['count_negative','count_positive']:
            if len(reward_para[0]) == 0:
                true_stop_sig  = 1
            elif max(reward_para[0]) <0.1:
                true_stop_sig = 1
            else:
                true_stop_sig = 0
    
        if (predict_stop_sig + true_stop_sig) ==2: #right stop
            reward_q = 0
            terminal = 1
        elif predict_stop_sig > true_stop_sig: #should not stop but stop
            reward_q = -1
            terminal = 0
        elif predict_stop_sig < true_stop_sig: #should stop but not stop
            reward_q = -0.5
            terminal = 1
        else:    #eval  the action
            c_before = np.array(reward_para[0])
            c_after = np.array(reward_para[1])
            c_before_mean = np.mean(c_before)            
            c_after_mean = np.mean(c_after)
            reward_g = c_before_mean - c_after_mean / c_before_mean

            if ques_type in ['exist_negative','exist_positive']:
                reward_l = (np.min(c_before) - np.min(c_after)) / np.min(c_before)
                terminal = int(np.min(c_after)<0.1)
            else:
                reward_l = (np.max(c_before) - np.max(c_after)) / np.max(c_before)
                terminal = int(np.max(c_after)<0.1)

            if reward_type == 'global':
                reward_q = reward_g
            elif reward_type == 'local':
                reward_q = reward_l
            elif reward_type == 'global + local':
                reward_q = 0.5*reward_g + 0.5*reward_l
        
        reward = reward_weight*reward_e + (1-reward_weight)*reward_q
        return reward, terminal,reward_e,reward_q
            
            


    def close(self):
        """
            End the simulation
        """
        self.ur5.disconnect()

