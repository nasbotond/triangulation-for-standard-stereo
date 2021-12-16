// TriangulationForStandardStereo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <stdio.h>  
#include <opencv2/opencv.hpp>  
#include <vector>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "PLYWriter.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

Mat read_matrix_from_file(const string file_name, const int rows, const int cols)
{
    ifstream file_stream(file_name);

    Mat K = Mat::zeros(rows, cols, CV_64F);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double e;
            file_stream >> e;
            K.at<double>(i, j) = e;
        }
    }

    file_stream.close();

    return K;
}


const double threshold = 3.;
const int ransac_max_iteration = 5000;
const double ransac_confidence = 0.99;

int main()
{
    for (int i = 1; i < 6; i++)
    {
        Mat img1 = imread("left" + std::to_string(i) + ".png", IMREAD_COLOR); // 800x600
        Mat img2 = imread("right" + std::to_string(i) + ".png", IMREAD_COLOR); // 800x600

        if (img1.empty() || img2.empty())
        {
            cout << "Could not open or find the image!\n" << endl;
            return -1;
        }

        Mat img1_gray, img2_gray;
        cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
        cvtColor(img2, img2_gray, COLOR_BGR2GRAY);

        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        int minHessian = 1; //400;
        Ptr<SURF> detector = SURF::create(minHessian);
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute(img1_gray, noArray(), keypoints1, descriptors1);
        detector->detectAndCompute(img2_gray, noArray(), keypoints2, descriptors2);
        cout << "Number of keypoints1: " << keypoints1.size() << endl;
        cout << "Number of keypoints2: " << keypoints2.size() << endl;

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f; // 0.5f
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            // good_matches.push_back(knn_matches[i][0]);
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
               good_matches.push_back(knn_matches[i][0]);
            }
        }
        
        vector<pair<Point2f, Point2f> > pointPairs;
        vector<Point2d> points_img1, points_img2;

        // Calibration matrix
        Mat K = read_matrix_from_file("malaga.txt", 3, 3);
        double deltaX = K.at<double>(0,2);
        // cout << deltaX << endl;
        double deltaY = K.at<double>(1,2);
        // cout << deltaY << endl;

        for (size_t i = 0; i < good_matches.size(); i++)
        {
            // pair<Point2f, Point2f> currPts;
            // currPts.first = keypoints2[good_matches[i].trainIdx].pt;
            // currPts.second = keypoints1[good_matches[i].queryIdx].pt;
            // pointPairs.push_back(currPts);
            Point2d point1, point2;
            point1 = keypoints1[good_matches[i].queryIdx].pt;
            point2 = keypoints2[good_matches[i].trainIdx].pt;
            point1.x -= deltaX;
            point1.y = deltaY - point1.y;
            point2.x -= deltaX;
            point2.y = deltaY - point2.y;
            points_img1.push_back(point1);
            points_img2.push_back(point2);
        }    
        cout << "Number of correspondences: " << points_img1.size() << endl;

        /*
        //-- Draw matches
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
            Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //-- Show detected matches
        imshow("Good Matches", img_matches);

        waitKey();
        return 0;
        */

        // IMPLEMENT THE SPECIAL TRIANGULATION METHOD FOR STANDARD STEREO (save point cloud in .xyz or .ply format)
        
        double b = 120.0; // mm (12 cm)
        double f = 3.8; // mm
        double pixel = 166.67; // pixel/mm
        double u1, u2, v1, d;
        
        vector<Point3d> spatialPoints;
        vector<Point3i> colors;
        Point3i newColor;
        Point3d newSpatialPoint;

        for (int i = 0; i < points_img1.size(); i++)
        {            
            u1 = points_img1[i].x / pixel;
            u2 = points_img2[i].x / pixel;
            v1 = points_img1[i].y / pixel;
            d = u1 - u2;

            newSpatialPoint.z = (b * f) / d / 100.0;
            newSpatialPoint.x = (-b * (u1 + u2)) / (2.0 * d) / 100.0;
            newSpatialPoint.y = (b * v1) / d / 100.0;

            if (abs(newSpatialPoint.z) <= 500.0 && newSpatialPoint.z >= 0)
            {
                spatialPoints.push_back(newSpatialPoint);

                newColor.x = 255;
                newColor.y = 0;
                newColor.z = 0;
                colors.push_back(newColor);
            }
        }

        // add origin in blue (for reference)
        Point3d origin (0.0, 0.0, 0.0);
        spatialPoints.push_back(origin);

        Point3i originColor(0, 0, 255);
        colors.push_back(originColor);

        // Write results into a PLY file. 
        // It can be visualized by open-source 3D application Meshlab (www.meshlab.org)
        std::string s = std::to_string(i);

        char outputFile[12];
        strcpy_s(outputFile, "result");
        strcat_s(outputFile, s.c_str());
        strcat_s(outputFile, ".ply");

        WritePLY(outputFile, spatialPoints, colors);      
    }
}
#else
int main()
{
    std::cout << "This code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
