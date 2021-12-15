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

#include "stereo_vision.h"

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
        int minHessian = 400;
        Ptr<SURF> detector = SURF::create(minHessian);
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute(img1_gray, noArray(), keypoints1, descriptors1);
        detector->detectAndCompute(img2_gray, noArray(), keypoints2, descriptors2);

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.5f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        
        vector<pair<Point2f, Point2f> > pointPairs;
        vector<Point2d> points_img1, points_img2;

        double deltaX = 400.0;
        double deltaY = 300.0;

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
            point1.y -= deltaY;
            point2.x -= deltaX;
            point2.y -= deltaY;
            points_img1.push_back(point1);
            points_img2.push_back(point2);
        }    

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

        for (int i = 0; i < points_img1.size(); i++)
        {
            Point3d newSpatialPoint;
            u1 = points_img1[i].x / pixel;
            u2 = points_img2[i].x / pixel;
            v1 = points_img1[i].y / pixel;
            d = u1 - u2;

            newSpatialPoint.z = (b * f) / d;
            newSpatialPoint.x = (-b * (u1 + u2)) / (2.0 * d);
            newSpatialPoint.y = (b * v1) / d;

            spatialPoints.push_back(newSpatialPoint);

            newColor.x = 255;
            newColor.y = 0;
            newColor.z = 0;
            colors.push_back(newColor);
        }

        // Write results into a PLY file. 
        // It can be visualized by open-source 3D application Meshlab (www.meshlab.org)
        std::string s = std::to_string(i);

        char outputFile[12];
        strcpy_s(outputFile, "result");
        strcat_s(outputFile, s.c_str());
        strcat_s(outputFile, ".ply");

        WritePLY(outputFile, spatialPoints, colors);

        /*
        // Normalize points
        std::cout << "Normalize points..." << std::endl;
        vector<Point2d> normalized_points_img1, normalized_points_img2;
        Mat T_1, T_2;
        stereo_vision::normalize_points(points_img1, normalized_points_img1, T_1);
        stereo_vision::normalize_points(points_img2, normalized_points_img2, T_2);

        // Find fundamental matrix
        cout << "Finding fundamental matrix..." << endl;
        Mat F_normalized;
        vector<int> inliers;

        // Since we use the normalized coordinates, we have to normalize the ransac threshold
        // We scale the threshold with the average scales of the src and dest images
        // (another solution is to find the inliers according to original coordinates after denormalize the fundamental matrix)

        double normalized_threshold = threshold * ((T_1.at<double>(0, 0) + T_2.at<double>(0, 0)) / 2.);
        cout << "ransac threshold : " << normalized_threshold << endl;

        stereo_vision::ransac_fundamental_matrix_8points_method(
            normalized_points_img1,
            normalized_points_img2,
            F_normalized,
            inliers,
            ransac_max_iteration,      // max iterations
            normalized_threshold,      // ransac threshold
            ransac_confidence          // ransac confidence
        );

        // Local optimization
        stereo_vision::get_fundamental_matrix_LSQ(
            normalized_points_img1,
            normalized_points_img2,
            inliers,
            F_normalized
        );

        cout << "Number of inliers: " << inliers.size() << endl;
        cout << "F_normalized: " << endl
            << F_normalized << endl;

        Mat F = T_2.t() * F_normalized * T_1;
        cout << "F:" << endl
            << F << endl;

        // Calibration matrix
        Mat K = read_matrix_from_file("malaga.txt", 3, 3);

        // Essential matrix
        Mat E = K.t() * F * K;
        //
        Mat R1, R2, t;
        stereo_vision::decompose_essential_matrix(
            E,
            R1,
            R2,
            t
        );


        Mat P1, P2, R, correct_t;
        stereo_vision::get_correct_rotation_translation(
            points_img1,
            points_img2,
            inliers,
            K,
            R1,
            R2,
            t,
            R,
            correct_t,
            P1,
            P2
        );
        */

        /*
        Mat H = EstimateHRANSAC(pointPairs, 0.2, 500);

        Mat transformedImage = Mat::zeros(1.5 * img1.size().height, 2.0 * img1.size().width, img1.type());
        transformImage(img1, transformedImage, Mat::eye(3, 3, CV_32F), true);
        transformImage(img2, transformedImage, H, true);

        imwrite("stitchedImage_" + std::to_string(i) + ".png", transformedImage);
        */
    }

    /*
    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    // imshow("Display window", transformedImage); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window

    return 0;
    */
}
#else
int main()
{
    std::cout << "This code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
