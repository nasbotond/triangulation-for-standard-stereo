# triangulation-for-standard-stereo

Given two photos from a camera setup that is a ‘standard stereo', this program retrieves the 3D point cloud of the environment.  
The program can be broken down in to the following subtasks: 

    Feature matching by OpenCV.
    Discard the outlier matchings using a robust method, the well-known RANSAC method.
    Implement the special triangulation method for standard stereo. 
    Save the obtained point cloud in XYZ or PLY format.  

The camera intrinsic matrix can be found in malaga.mat. (The parameters are the same for the cameras.) The baseline is 12cm.