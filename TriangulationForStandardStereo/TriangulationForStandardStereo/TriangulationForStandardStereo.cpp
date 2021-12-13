// TriangulationForStandardStereo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>  
#include <opencv2/opencv.hpp>  
#include <vector>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

int main()
{
    for (int i = 1; i < 6; i++)
    {
        Mat img1 = imread("left" + std::to_string(i) + ".jpg", IMREAD_COLOR);
        Mat img2 = imread("right" + std::to_string(i) + ".jpg", IMREAD_COLOR);

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

        for (size_t i = 0; i < good_matches.size(); i++)
        {
            pair<Point2f, Point2f> currPts;
            currPts.first = keypoints2[good_matches[i].trainIdx].pt;
            currPts.second = keypoints1[good_matches[i].queryIdx].pt;
            pointPairs.push_back(currPts);
        }

        // IMPLEMENT THE SPECIAL TRIANGULATION METHOD FOR STANDARD STEREO (save point cloud in .xyz or .ply format)

        /*
        Mat H = EstimateHRANSAC(pointPairs, 0.2, 500);

        Mat transformedImage = Mat::zeros(1.5 * img1.size().height, 2.0 * img1.size().width, img1.type());
        transformImage(img1, transformedImage, Mat::eye(3, 3, CV_32F), true);
        transformImage(img2, transformedImage, H, true);

        imwrite("stitchedImage_" + std::to_string(i) + ".png", transformedImage);
        */
    }

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    // imshow("Display window", transformedImage); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window

    return 0;
}
#else
int main()
{
    std::cout << "This code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
