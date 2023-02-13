import json
import os, sys
from os.path import join
import numpy as np

from .frame_io import read_frame_graph
from .shape_struct import ShapeStruct

class MetadataReader(object):

    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = data_dir
        self.frame_graph = None

        # Member variables that will be assigned later.
        self.metadata = None

        self.num_cams = None
        self.cam_paths_list = None
        self.cam_to_poses_dict = None
        self.cam_to_camdata = None

        self.rig_is_cam = None
        self.rig_path = None
        self.rig_paths_list = None

        self.init_cam_list = None
        self.init_imgtype_list = None
    
    def read_metadata_and_initialize_dirs(self, metadata_path, frame_graph_path, create_dirs=True):
        '''
        Reads in the specified metadata file, which sets important variables such as number of cameras and their extrinsics.
        Also sets up the directory structure according to the specified metadata.
        
        creawte_dirs: If True, creates the associated directory structure. 
        '''
        
        # Read the frame graph first.
        self.frame_graph = read_frame_graph(frame_graph_path)
        print('Frame graph read successfully. ')
        
        with open(metadata_path) as metadata_file:

            #Load Metadata JSON and set the number of cameras
            self.metadata = json.load(metadata_file)
            self.num_cams = len(self.metadata['cams'])

            #Initialize indexing lists  
            self.cam_paths_list = []
            self.rig_paths_list = []
            self.cam_to_poses_dict = dict()

            #Print the number of found cameras
            print(f"Number of cameras found... {self.num_cams}!")

            #Initialize the camera name/id to camera data dictionary. 
            #This is used during data collection to streamline the data collection procedures.
            self.cam_to_camdata = dict()

            #Make a rig directory and initialize the rigdata struct.
            # self.rig_path = join(self.data_dir, "rig")
            self.rig_path = "rig"
            rig_out_dir = join(self.data_dir, "rig")
            if create_dirs:
                if not os.path.exists(rig_out_dir):
                    os.makedirs(rig_out_dir)

            rigdata = dict(
                path=rig_out_dir,
                types=self.metadata["rig_img_types"],
                is_rig=True,
                data=dict(frame="rbf")
            )

            #Initialize camera headers and the rig_is_cam flag. The rig_is_cam flag is used if 
            #a camera has its position at the origin, which is where the rig frame is located.
            #If this is true, then the first camera and only that camera will be indexed as the
            #rig camera and a new image will not be created for the rig camera.
            cam_headers = []
            self.rig_is_cam = False

            #Iterate through each found camera...
            for i, c in enumerate(self.metadata['cams']):
                
                #For each camera, create a directory and index that directory in the csv index.
                c_str = f"cam{i}"
                cpath = join(self.data_dir, c_str)
                cam_headers.append(c_str)
                if create_dirs:
                    if not os.path.exists(cpath):
                        os.makedirs(cpath)

                #Also create a camera data struct that holds all important data for data collection
                cdata = dict(
                    path=cpath,
                    types=c["img_types"],
                    data=c
                )

                # Get the pose of the camera by querying the frame graph.
                frame_name = c["frame"]
                T_rig_cam = self.frame_graph.query_transform(f0="rbf", f1=frame_name) # FTensor0
                cam_position = T_rig_cam.translation.cpu().numpy()

                print("METADATAREADER-----")
                print(frame_name, cam_position)
                # cam_orientation = T_rig_cam.rotation.cpu().numpy()

                #If a camera is the first to have an origin position at the rig frame, set the rig_is_cam
                #flag to be true. Also include the rigdata in the list of data associated with the camera number.
                #If it is not the first camera, warn the user. Otherwise, add the camera data to the index.
                # if np.array_equal(np.array(c["pos"]),np.array([0.0,0.0,0.0])):
                if np.array_equal(cam_position,np.array([0.0,0.0,0.0])):
                    if self.rig_is_cam:
                        print(f"Camera {i} also is positioned at the origin (Numbering starts at 0). \
                              Since a previous camera was also positioned at the rig frame, this camera will \
                              not be indexed as the rig camera. To ensure that camera {i} is the rig camera, \
                              change the order of the camera in the metadata to be the first camera.")
                        continue
                    else:
                        self.rig_is_cam = True

                        cdata["types"] = list(set(cdata["types"]+rigdata["types"]))
                        cdata.update({"is_rig":True})
                        self.cam_to_camdata.update({
                            i:cdata
                        })

                        # self.rig_path = cdata["path"]
                        self.rig_path = c_str
                    
                self.cam_to_camdata.update({
                    i:cdata
                })

                self.cam_to_poses_dict.update({
                    i:list()
                })

            #If the rig_is_cam flag was not set to true, then add the rig as a seperate camera to the dictionary.
            #This indicates that none of the cameras are located at the rig frame. Also add the rigpath to the 
            #csv index.
            if not self.rig_is_cam:
                self.cam_to_camdata.update({
                    "rig":rigdata
                })

                self.cam_to_poses_dict.update({
                    "rig":list()
                })

                cam_headers.append(self.rig_path)

            #Place the headers at the top of the csv index
            self.cam_paths_list.append(cam_headers)
            self.rig_paths_list.append(self.metadata["rig_img_types"])

            print(self.rig_paths_list)

            #To initialize the ImageClient, a default cam list and image type is needed.
            #The image types can be changed on-the-fly later in the pipeline.
            self.init_cam_list = self.metadata["cams"][0]["airsim_cam_nums"]
            self.init_imgtype_list = self.metadata["cams"][0]["img_types"]
    
            #Print the camera to data conversion dictionary to ensure everything was read correctly.
            print(self.cam_to_camdata)