#ifndef SFM_GENERAL_STEREO
#define SFM_GENERAL_STEREO

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <fstream>



namespace stereo_vision
{

	void read_matching_file(
		const std::string matching_file_name,
		std::vector<cv::Point2d>& src_points,
		std::vector<cv::Point2d>& dest_points)
	{
		std::ifstream file(matching_file_name);

		int n;
		file >> n;

		for (int i = 0; i < n; i++)
		{
			double u1, v1, u2, v2;
			file >> u1 >> v1 >> u2 >> v2;

			src_points.push_back(cv::Point2d(u1, v1));
			dest_points.push_back(cv::Point2d(u2, v2));
		}

		file.close();
	}

	void draw_correspondences(
		const cv::Mat& image1,
		const std::vector<cv::Point2d>& points1,
		const cv::Mat& image2,
		const std::vector<cv::Point2d>& points2,
		const std::vector<int>& inliers,
		cv::Mat& output)
	{
		std::vector<cv::KeyPoint> keypoints1, keypoints2;
		std::vector<cv::DMatch> mt;

		int n = inliers.size();
		for (int i = 0; i < n; i++)
		{
			int idx = inliers[i];
			keypoints1.push_back(cv::KeyPoint(points1[idx], 1.));
			keypoints2.push_back(cv::KeyPoint(points2[idx], 1.));

			cv::DMatch dmatch;
			dmatch.queryIdx = keypoints1.size() - 1;
			dmatch.trainIdx = dmatch.queryIdx;
			mt.push_back(dmatch);
		}

		cv::drawMatches(image1, keypoints1, image2, keypoints2, mt, output);
	}

	static void draw_correspondences(
		const cv::Mat& image1,
		const std::vector<cv::Point2d>& points1,
		const cv::Mat& image2,
		const std::vector<cv::Point2d>& points2,
		cv::Mat& output)
	{
		std::vector<cv::KeyPoint> keypoints1, keypoints2;
		std::vector<cv::DMatch> mt;

		int n = points1.size();
		for (int idx = 0; idx < n; idx++)
		{
			keypoints1.push_back(cv::KeyPoint(points1[idx], 1.));
			keypoints2.push_back(cv::KeyPoint(points2[idx], 1.));

			cv::DMatch dmatch;
			dmatch.queryIdx = keypoints1.size() - 1;
			dmatch.trainIdx = dmatch.queryIdx;
			mt.push_back(dmatch);
		}

		cv::drawMatches(image1, keypoints1, image2, keypoints2, mt, output);
	}

	cv::Mat skew_symmetric(const cv::Mat& t)
	{
		cv::Mat T = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2), t.at<double>(1),
			t.at<double>(2), 0, -t.at<double>(0),
			-t.at<double>(1), t.at<double>(0), 0);

		return T;
	}

	/**
	 * translating the mass point to the origin and
	 * the average distance from the mass point to be sqrt(2).
	 *
	 * @param points input image
	 * @param normalized_points output normalized points
	 * @param T transformation matrix from orignal coordinates to the normalized one
	 */
	void normalize_points(
		const std::vector<cv::Point2d>& points,
		std::vector<cv::Point2d>& normalized_points,
		cv::Mat& T)
	{
		int n = points.size();

		normalized_points.resize(n);

		// Calculate the mass point
		cv::Point2d mass(0, 0);

		for (auto i = 0; i < n; ++i)
		{
			mass = mass + points[i];
		}

		mass = mass * (1.0 / n);

		// Translate the point clouds to the origin
		for (auto i = 0; i < n; ++i)
		{
			normalized_points[i] = points[i] - mass;
		}

		// Calculate the average distances of the points from the origin
		double avg_distance = 0.0;

		for (auto i = 0; i < n; ++i)
		{
			avg_distance += cv::norm(normalized_points[i]);
		}

		avg_distance /= n;

		const double scalar =
			sqrt(2) / avg_distance;

		for (auto i = 0; i < n; ++i)
		{
			normalized_points[i] *= scalar;
		}

		T = cv::Mat::eye(3, 3, CV_64F);
		T.at<double>(0, 0) = scalar;
		T.at<double>(1, 1) = scalar;
		T.at<double>(0, 2) = -scalar * mass.x;
		T.at<double>(1, 2) = -scalar * mass.y;
	}

	void get_fundamental_matrix_LSQ(
		const std::vector<cv::Point2d>& src_points,
		const std::vector<cv::Point2d>& dest_points,
		std::vector<int>& selected_samples,
		cv::Mat& fundamental_matrix)
	{
		int n = selected_samples.size();
		// Construct the coefficient matrix (A)
		cv::Mat A(n, 9, CV_64F);

		for (int i = 0; i < n; i++)
		{
			int idx = selected_samples[i];

			const double
				& x1 = src_points[idx].x,
				& y1 = src_points[idx].y,
				& x2 = dest_points[idx].x,
				& y2 = dest_points[idx].y;

			A.at<double>(i, 0) = x1 * x2;
			A.at<double>(i, 1) = x2 * y1;
			A.at<double>(i, 2) = x2;
			A.at<double>(i, 3) = y2 * x1;
			A.at<double>(i, 4) = y2 * y1;
			A.at<double>(i, 5) = y2;
			A.at<double>(i, 6) = x1;
			A.at<double>(i, 7) = y1;
			A.at<double>(i, 8) = 1;
		}

		// Solve Ax=0 where x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
		cv::Mat evals, evecs;
		cv::Mat AtA = A.t() * A;
		cv::eigen(AtA, evals, evecs);

		cv::Mat x = evecs.row(evecs.rows - 1);
		fundamental_matrix.create(3, 3, CV_64F);
		memcpy(fundamental_matrix.data, x.data, sizeof(double) * 9);
	}

	int get_iteration_number(const int point_number,
		const int inlier_number,
		const int sample_size,
		const double confidence)
	{
		const double inlier_ratio =
			static_cast<double>(inlier_number) / point_number;

		static const double log1 = log(1.0 - confidence);
		const double log2 = log(1.0 - pow(inlier_ratio, sample_size));

		const int k = log1 / log2;
		if (k < 0)
			return std::numeric_limits<int>::max();
		return k;
	}

	/**
	 * Get fundmental matrix using 8 points method and RANSAC roubstification
	 */
	void ransac_fundamental_matrix_8points_method(
		const std::vector<cv::Point2d>& src_points,
		const std::vector<cv::Point2d>& dest_points,
		cv::Mat& best_fundamental_matrix,
		std::vector<int>& best_inliers,
		const int ransac_max_iteration,
		const double ransac_threshold,
		const double ransac_confidence)
	{
		// Number of correspondences
		const int n = src_points.size();

		// Size of a minimal sample
		const int sample_size = 8;

		int iteration = 0;
		int maximum_iterations = std::numeric_limits<int>::max(); // The maximum number of iterations set adaptively when a new best model is found
		while (iteration++ < std::min(ransac_max_iteration, maximum_iterations))
		{
			// Initializing the index pool from which the minimal samples are selected
			cv::Mat mask = cv::Mat::ones(n, 1, CV_8U);
			// The minimal sample
			std::vector<int> selected_points_idx;

			// Select 8 random correspondeces
			do
			{
				// Select a random index
				int idx = rand() % n;

				// In case it is not selected before
				if (mask.at<uchar>(idx))
				{
					// new index is selected
					mask.at<uchar>(idx) = 0;
					selected_points_idx.push_back(idx);
				}

			} while (selected_points_idx.size() != sample_size);

			// Estimate fundamental matrix
			cv::Mat fundamental_matrix(3, 3, CV_64F);
			get_fundamental_matrix_LSQ(
				src_points,
				dest_points,
				selected_points_idx,
				fundamental_matrix);

			// Count the inliers
			std::vector<int> inliers;
			for (int i = 0; i < src_points.size(); ++i)
			{
				// Symmetric epipolar distance
				cv::Mat pt1 = (cv::Mat_<double>(3, 1) << src_points[i].x, src_points[i].y, 1);
				cv::Mat pt2 = (cv::Mat_<double>(3, 1) << dest_points[i].x, dest_points[i].y, 1);

				// Calculate the error
				cv::Mat lL = fundamental_matrix.t() * pt2;
				cv::Mat lR = fundamental_matrix * pt1;

				// Calculate the distance of point pt1 from lL
				const double
					& aL = lL.at<double>(0),
					& bL = lL.at<double>(1),
					& cL = lL.at<double>(2);

				double tL = abs(aL * src_points[i].x + bL * src_points[i].y + cL);
				double dL = sqrt(aL * aL + bL * bL);
				double distanceL = tL / dL;

				// Calculate the distance of point pt2 from lR
				const double
					& aR = lR.at<double>(0),
					& bR = lR.at<double>(1),
					& cR = lR.at<double>(2);

				double tR = abs(aR * dest_points[i].x + bR * dest_points[i].y + cR);
				double dR = sqrt(aR * aR + bR * bR);
				double distanceR = tR / dR;

				double dist = 0.5 * (distanceL + distanceR);

				if (dist < ransac_threshold)
					inliers.push_back(i);
			}

			// Update if the new model is better than the previous so-far-the-best.
			if (best_inliers.size() < inliers.size())
			{
				// Update the set of inliers
				best_inliers.clear();
				best_inliers.assign(inliers.begin(), inliers.end());
				// Update fundamental matrix
				fundamental_matrix.copyTo(best_fundamental_matrix);
				// Update the iteration number
				maximum_iterations = get_iteration_number(n,
					best_inliers.size(),
					sample_size,
					ransac_confidence);
			}
		}
	}

	void decompose_essential_matrix(
		const cv::Mat& E,
		cv::Mat& R1,
		cv::Mat& R2,
		cv::Mat& t)
	{
		cv::SVD svd(E, cv::SVD::FULL_UV);
		// It gives matrices U D Vt

		if (cv::determinant(svd.u) < 0)
			svd.u.col(2) *= -1;
		if (cv::determinant(svd.vt) < 0)
			svd.vt.row(2) *= -1;

		cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0,
			1, 0, 0,
			0, 0, 1);

		cv::Mat rotation_1 = svd.u * w * svd.vt;
		cv::Mat rotation_2 = svd.u * w.t() * svd.vt;
		cv::Mat translation = svd.u.col(2) / cv::norm(svd.u.col(2));
		rotation_1.copyTo(R1);
		rotation_2.copyTo(R2);
		translation.copyTo(t);
	}

	void linear_triangulation(
		const cv::Mat& projection_1,
		const cv::Mat& projection_2,
		const cv::Point2d& src_point,
		const cv::Point2d& dst_point,
		cv::Point3d& points3d)
	{
		cv::Mat A(4, 3, CV_64F);
		cv::Mat b(4, 1, CV_64F);

		{
			const double
				& px = src_point.x,
				& py = src_point.y,
				& p1 = projection_1.at<double>(0, 0),
				& p2 = projection_1.at<double>(0, 1),
				& p3 = projection_1.at<double>(0, 2),
				& p4 = projection_1.at<double>(0, 3),
				& p5 = projection_1.at<double>(1, 0),
				& p6 = projection_1.at<double>(1, 1),
				& p7 = projection_1.at<double>(1, 2),
				& p8 = projection_1.at<double>(1, 3),
				& p9 = projection_1.at<double>(2, 0),
				& p10 = projection_1.at<double>(2, 1),
				& p11 = projection_1.at<double>(2, 2),
				& p12 = projection_1.at<double>(2, 3);

			A.at<double>(0, 0) = px * p9 - p1;
			A.at<double>(0, 1) = px * p10 - p2;
			A.at<double>(0, 2) = px * p11 - p3;
			A.at<double>(1, 0) = py * p9 - p5;
			A.at<double>(1, 1) = py * p10 - p6;
			A.at<double>(1, 2) = py * p11 - p7;

			b.at<double>(0) = p4 - px * p12;
			b.at<double>(1) = p8 - py * p12;
		}

		{
			const double
				& px = dst_point.x,
				& py = dst_point.y,
				& p1 = projection_2.at<double>(0, 0),
				& p2 = projection_2.at<double>(0, 1),
				& p3 = projection_2.at<double>(0, 2),
				& p4 = projection_2.at<double>(0, 3),
				& p5 = projection_2.at<double>(1, 0),
				& p6 = projection_2.at<double>(1, 1),
				& p7 = projection_2.at<double>(1, 2),
				& p8 = projection_2.at<double>(1, 3),
				& p9 = projection_2.at<double>(2, 0),
				& p10 = projection_2.at<double>(2, 1),
				& p11 = projection_2.at<double>(2, 2),
				& p12 = projection_2.at<double>(2, 3);

			A.at<double>(2, 0) = px * p9 - p1;
			A.at<double>(2, 1) = px * p10 - p2;
			A.at<double>(2, 2) = px * p11 - p3;
			A.at<double>(3, 0) = py * p9 - p5;
			A.at<double>(3, 1) = py * p10 - p6;
			A.at<double>(3, 2) = py * p11 - p7;

			b.at<double>(2) = p4 - px * p12;
			b.at<double>(3) = p8 - py * p12;
		}

		//cv::Mat x = (A.t() * A).inv() * A.t() * b;
		cv::Mat x = A.inv(cv::DECOMP_SVD) * b;
		points3d = cv::Point3d(x);
	}

	int check_points_infront_of_camera(
		const std::vector<cv::Point2d>& src_points,
		const std::vector<cv::Point2d>& dest_points,
		const std::vector<int>& inliers,
		const cv::Mat& P0,
		const cv::Mat& P1)
	{
		int n = inliers.size();

		int infront_camera_cnt = 0;
		double average_reprojection_error = 0;
		for (int i = 0; i < n; i++)
		{
			int idx = inliers[i];

			// Triangulate the point
			cv::Point3d point3d;
			linear_triangulation(P0, P1, src_points[idx], dest_points[idx], point3d);

			// infront of the camera
			if (point3d.z > 0)
				infront_camera_cnt++;

			// calculate the reporjection error
			cv::Mat projection1 = P0 * (cv::Mat_<double>(4, 1) << point3d.x, point3d.y, point3d.z, 1.);
			cv::Mat projection2 = P1 * (cv::Mat_<double>(4, 1) << point3d.x, point3d.y, point3d.z, 1.);
			projection1 /= projection1.at<double>(2);
			projection2 /= projection2.at<double>(2);

			// cv::norm(projection1 - src_point_)
			double dx1 = projection1.at<double>(0) - src_points[idx].x;
			double dy1 = projection1.at<double>(1) - src_points[idx].y;
			double squaredDist1 = dx1 * dx1 + dy1 * dy1;

			// cv::norm(projection2 - dst_point_)
			double dx2 = projection2.at<double>(0) - dest_points[idx].x;
			double dy2 = projection2.at<double>(1) - dest_points[idx].y;
			double squaredDist2 = dx2 * dx2 + dy2 * dy2;

			double dst = (sqrt(squaredDist1) + sqrt(squaredDist2)) / 2.;
			//std::cout << dst << std::endl;
			average_reprojection_error += dst;
		}
		average_reprojection_error /= n;
		std::cout << "average_reprojection_error: " << average_reprojection_error << std::endl;

		return infront_camera_cnt;
	}

	void SaveAlgebraicTriangulatedPoints(
		const std::vector<cv::Point2d>& src_points,
		const std::vector<cv::Point2d>& dest_points,
		const std::vector<int>& inliers,
		const cv::Mat& P0,
		const cv::Mat& P1)
	{
		int n = inliers.size();
		std::ofstream file("AlgebraicTriangulatedPoints.xyz");
		for (int i = 0; i < n; i++)
		{
			int idx = inliers[i];
			cv::Point3d point3d;
			linear_triangulation(P0, P1, src_points[idx], dest_points[idx], point3d);
			file << point3d.x << " " << point3d.y << " " << point3d.z << std::endl;
		}
		file.close();
	}

	void MidPointTriangulation(
		const cv::Point2d& u1,
		const cv::Point2d& u2,
		const cv::Mat& R,
		const cv::Mat& t,
		const cv::Mat& K,
		cv::Point3d& triangulatedPoint)
	{
		cv::Point3d u1H(u1.x, u1.y, 1.0);
		cv::Point3d u2H(u2.x, u2.y, 1.0);

		cv::Mat KInv = K.inv();

		cv::Mat dir1 = KInv * cv::Mat(u1H);

		cv::Mat Rt;
		cv::transpose(R, Rt);

		cv::Mat dir2 = t + Rt * ((KInv* (cv::Mat(u2H)))-t);
		cv::Mat dir3 = dir1.cross(dir2);

		cv::Mat A(3, 3, CV_64F);
		A.at<double>(0, 0) = dir1.at<double>(0);
		A.at<double>(0, 1) = dir2.at<double>(0);
		A.at<double>(0, 2) = dir3.at<double>(0);
		A.at<double>(1, 0) = dir1.at<double>(1);
		A.at<double>(1, 1) = dir2.at<double>(1);
		A.at<double>(1, 2) = dir3.at<double>(1);
		A.at<double>(2, 0) = dir1.at<double>(2);
		A.at<double>(2, 1) = dir2.at<double>(2);
		A.at<double>(2, 2) = dir3.at<double>(2);

		cv::Mat params = A.inv() * t;

		cv::Mat triangulatedPointMat = -params.at<double>(0) * dir1 - params.at<double>(2)/2.0 * dir3;

		triangulatedPoint.x = triangulatedPointMat.at<double>(0);
		triangulatedPoint.y = triangulatedPointMat.at<double>(1);
		triangulatedPoint.z = triangulatedPointMat.at<double>(2);
	}

	void SaveMidPointTriangulatedPoints(
		const std::vector<cv::Point2d>& src_points,
		const std::vector<cv::Point2d>& dest_points,
		const std::vector<int>& inliers,
		const cv::Mat& R,
		const cv::Mat& t,
		const cv::Mat& K)
	{
		int n = inliers.size();
		std::ofstream file("MidPointTriangulatedPoints.xyz");
		std::cout << "SaveMidPointTriangulatedPoints:" << std::endl;
		for (int i = 0; i < n; i++)
		{
			int idx = inliers[i];
			cv::Point3d point3d;
			MidPointTriangulation(src_points[idx], dest_points[idx],R,t,K, point3d);
			file << point3d.x << " " << point3d.y << " " << point3d.z << std::endl;
		}
		file.close();
	}

	void get_correct_rotation_translation(
		const std::vector<cv::Point2d>& src_points,
		const std::vector<cv::Point2d>& dest_points,
		const std::vector<int>& inliers,
		const cv::Mat& K,
		const cv::Mat& R1,
		const cv::Mat& R2,
		const cv::Mat& t,
		cv::Mat& correct_R,
		cv::Mat& correct_t,
		cv::Mat& projection_mat0,
		cv::Mat& projection_mat1)
	{
		cv::Mat Rs[] = { R1, R2, R1, R2 };
		cv::Mat ts[] = { t, t, -t, -t };

		// Consider the first camera is the origin
		cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);

		int best_infront_of_camera = 0;
		int best_idx;
		for (int i = 0; i < 4; i++)
		{
			// P1 = K * [R|t]
			cv::Mat P1(3, 4, CV_64F);
			P1(cv::Range(0, 3), cv::Range(0, 3)) = 1.0 * Rs[i];
			P1.col(3) = 1.0 * ts[i];
			P1 = K * P1;

			const int points_infront_camera = check_points_infront_of_camera(
				src_points,
				dest_points,
				inliers,
				P0,
				P1);
			std::cout << "points infront of the camera: " << points_infront_camera << "/" << inliers.size() << std::endl;

			if (points_infront_camera > best_infront_of_camera)
			{
				best_infront_of_camera = points_infront_camera;
				best_idx = i;
			}
		}

		std::cout << "best_idx: " << best_idx << std::endl;

		cv::Mat P1(3, 4, CV_64F);
		P1(cv::Range(0, 3), cv::Range(0, 3)) = 1.0 * Rs[best_idx];
		P1.col(3) = 1.0 * ts[best_idx];
		P1 = K * P1;
		SaveAlgebraicTriangulatedPoints(src_points, dest_points, inliers, P0, P1);

		Rs[best_idx].copyTo(correct_R);
		ts[best_idx].copyTo(correct_t);

		SaveMidPointTriangulatedPoints(src_points, dest_points, inliers, correct_R, correct_t,K);

		P0.copyTo(projection_mat0);
		projection_mat1 = cv::Mat(3, 4, CV_64F);
		projection_mat1(cv::Range(0, 3), cv::Range(0, 3)) = 1.0 * correct_R;
		projection_mat1.col(3) = 1.0 * correct_t;
		projection_mat1 = K * projection_mat1;
	}
}

#endif