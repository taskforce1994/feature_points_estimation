#pragma once
#include "sub_module.h"

int get_video_frame(string path, vector<Mat>& rgb_frame) {

	//Get Video Frame
	printf("Get Video Frame\n");

	int frame_number = 0;
	VideoCapture capture(path);
	Mat frame;

	if (!capture.isOpened()) {
		printf("File can not open.\n");
		return 0;
	}
	while (1) {
		//grab frame from file & throw to Mat
		capture >> frame;
		if (frame.empty()) //Is video end?
			break;

		rgb_frame.push_back(frame.clone());
		//Display and delay
		//imshow("w", rgb_frame[frame_number]);
		//if (waitKey(10) > 0)
		//	break;
		frame_number++;
	}
	printf("Total frame num  %d \n", rgb_frame.size());

	return frame_number;
};

int get_video_frame_gray(string path, vector<Mat>& gray, vector<Mat>& rgb_frame) {

	//Get Video Frame
	printf("Get Video Frame\n");

	int frame_number = 0;
	VideoCapture capture(path);
	Mat frame;

	if (!capture.isOpened()) {
		printf("File can not open.\n");
		return 0;
	}
	while (1) {
		//grab frame from file & throw to Mat
		capture >> frame;
		if (frame.empty()) //Is video end?
			break;

		char path2[256];
		rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
		sprintf_s(path2, "./ori_frame/%d_frame.bmp", frame_number);
		imwrite(path2, frame);
		rgb_frame.push_back(frame.clone());
		cvtColor(frame, frame, COLOR_RGB2GRAY);
	
		gray.push_back(frame.clone());
		//Display and delay
		imshow("w", gray[frame_number]);
		if (waitKey(10) > 0)
			break;
		frame_number++;
	}
	printf("Total frame num  %d \n", gray.size());

	return frame_number;
};

int rgb_split(string write_path, vector<Mat>& r, vector<Mat>& g, vector<Mat>& b, vector<Mat>rgb_frame) {

	//Start RGB Split
	printf("\nStart RGB Split\n");
	for (int frame_num = 0; frame_num < rgb_frame.size(); frame_num += 1) {

		Mat bgr_buf[3];
		split(rgb_frame[frame_num], bgr_buf);

		b.push_back(bgr_buf[0]);
		g.push_back(bgr_buf[1]);
		r.push_back(bgr_buf[2]);

		string rgb_write_path = write_path +"/rgb/" + to_string(frame_num + 1) + ".rgb.bmp";
		string red_write_path = write_path + "/red/" + to_string(frame_num + 1) + ".r.bmp";
		string green_write_path = write_path + "/green/" + to_string(frame_num + 1) + ".g.bmp";
		string blue_write_path = write_path + "/blue/" + to_string(frame_num + 1) + ".b.bmp";

		imwrite(rgb_write_path, rgb_frame[frame_num]);
		imwrite(red_write_path, r[frame_num]);
		imwrite(green_write_path, g[frame_num]);
		imwrite(blue_write_path, b[frame_num]);

		bgr_buf[0].release();
		bgr_buf[1].release();
		bgr_buf[2].release();
	}
	printf("End RGB Split\n\n");
	//End RGB Split
}

int save_all_descriptor(string write_path, SIFT_parameter SIFT_param, vector<Mat> r, vector<Mat> g, vector<Mat> b, vector<vector<KeyPoint>>& all_keypoints, vector<Mat>& all_descriptors) {


	printf("Start SIFT and Save Descriptor\n\n");
	int total_frame = r.size();
	for (int frame_num = 0; frame_num < total_frame; frame_num += 1) {

		Ptr<SIFT> detector = SIFT::create
		(SIFT_param.feature_num, SIFT_param.octave, SIFT_param.contrast,
			SIFT_param.edge, SIFT_param.sigma);
		vector<KeyPoint> keypoints_r;	Mat descriptors_r;
		vector<KeyPoint> keypoints_g;	Mat descriptors_g;
		vector<KeyPoint> keypoints_b;	Mat descriptors_b;

		//Get RGB Feature points
		detector->detectAndCompute(r[frame_num], noArray(), keypoints_r, descriptors_r);
		all_keypoints.push_back(keypoints_r);
		all_descriptors.push_back(descriptors_r);
		detector->detectAndCompute(g[frame_num], noArray(), keypoints_g, descriptors_g);
		all_keypoints.push_back(keypoints_g);
		all_descriptors.push_back(descriptors_g);
		detector->detectAndCompute(b[frame_num], noArray(), keypoints_b, descriptors_b);
		all_keypoints.push_back(keypoints_b);
		all_descriptors.push_back(descriptors_b);


		string red_write_path = write_path + to_string(frame_num + 1) + ".r.bmp";
		string green_write_path = write_path + to_string(frame_num + 1) + ".g.bmp";
		string blue_write_path = write_path + to_string(frame_num + 1) + ".b.bmp";


		imwrite(red_write_path, all_descriptors[frame_num]);
		imwrite(green_write_path, all_descriptors[frame_num + 1]);
		imwrite(blue_write_path, all_descriptors[frame_num + 2]);

		printf("%d frame key points Red %d Green %d Blue %d\n"
			, frame_num + 1, all_keypoints[frame_num].size()
			, all_keypoints[frame_num + 1].size(), all_keypoints[frame_num + 2].size());

		descriptors_r.release();
		descriptors_g.release();
		descriptors_b.release();
		char path[256];
		Mat img_matches;
		drawKeypoints(r[frame_num], all_keypoints[3 * (frame_num)], img_matches);
		sprintf_s(path, "./keypoints/red.%d.bmp", frame_num);
		imwrite(path, img_matches);

		drawKeypoints(g[frame_num], all_keypoints[3 * (frame_num) + 1], img_matches);
		sprintf_s(path, "./keypoints/green.%d.bmp", frame_num );
		imwrite(path, img_matches);

		drawKeypoints(b[frame_num], all_keypoints[3 * (frame_num) + 2], img_matches);
		sprintf_s(path, "./keypoints/blue.%d.bmp", frame_num);
		imwrite(path, img_matches);
	}

	printf("End SIFT and Save Descriptor\n");


}
int save_all_descriptor_gray(string write_path, SIFT_parameter SIFT_param, vector<Mat> y, vector<vector<KeyPoint>>& all_keypoints, vector<Mat>& all_descriptors, int frame_interval) {


	printf("Start SIFT and Save Descriptor\n\n");
	int total_frame = y.size();
	for (int frame_num = 0; frame_num < total_frame; frame_num += 1) {
		if (frame_num % frame_interval == 0) {
			Ptr<SIFT> detector = SIFT::create
			(SIFT_param.feature_num, SIFT_param.octave, SIFT_param.contrast,
				SIFT_param.edge, SIFT_param.sigma);
			vector<KeyPoint> keypoints_y;	Mat descriptors_y;
			//Get RGB Feature points
			detector->detectAndCompute(y[frame_num], noArray(), keypoints_y, descriptors_y);
			all_keypoints.push_back(keypoints_y);
			all_descriptors.push_back(descriptors_y);
			printf("%d frame key points Gray %d \n"
				, frame_num + 1, all_keypoints[frame_num].size());

			descriptors_y.release();
		
		}
		else
		{
			vector<KeyPoint> keypoints_y ;	Mat descriptors_y;
			all_keypoints.push_back(all_keypoints[frame_num-1]);
			all_descriptors.push_back(all_descriptors[frame_num-1]);
		}
		string gray_write_path = write_path + to_string(frame_num + 1) + ".gray.bmp";
		imwrite(gray_write_path, all_descriptors[frame_num]);
		char path[256];
		Mat img_matches;
		drawKeypoints(y[frame_num], all_keypoints[frame_num], img_matches);
		sprintf_s(path, "./keypoints/gray.%d.bmp", frame_num);
		imwrite(path, img_matches);
	}

	printf("End SIFT and Save Descriptor\n");


}

int matching_feature_points(Matching_parameter Matching_param, vector<vector<KeyPoint>> all_keypoints, vector<Mat>& all_descriptors, vector<Point2f>& matching_point_before, vector<Point2f>& matching_point_after, vector<DMatch> &matching_feature, int frame_num, int frame_interval, int color) {
	
	color_name color_str;

	if (color ==0 || color == 1|| color == 2) {
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
		vector< vector<DMatch> > knn_matches;
		matcher->knnMatch(all_descriptors[3 * (frame_num - frame_interval) + color].clone(), all_descriptors[3 * frame_num + color].clone(), knn_matches, 2);

		for (size_t i = 0; i < knn_matches.size(); i++)
			if (knn_matches[i][0].distance < Matching_param.ratio_thresh * knn_matches[i][1].distance)
				matching_feature.push_back(knn_matches[i][0]);

		for (size_t i = 0; i < matching_feature.size(); i++) {
			matching_point_before.push_back(all_keypoints[3 * (frame_num - frame_interval) + color][matching_feature[i].queryIdx].pt);
			matching_point_after.push_back(all_keypoints[3 * frame_num + color][matching_feature[i].trainIdx].pt);
		}

		printf("%s  Matched feature points : %d\n", color_str.color_str[color], matching_feature.size());
	}
	else
	{
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		vector< vector<DMatch> > knn_matches;
		matcher->knnMatch(all_descriptors[frame_num - frame_interval].clone(), all_descriptors[frame_num].clone(), knn_matches, 2);

		for (size_t i = 0; i < knn_matches.size(); i++)
			if (knn_matches[i][0].distance < Matching_param.ratio_thresh * knn_matches[i][1].distance)
				matching_feature.push_back(knn_matches[i][0]);

		for (size_t i = 0; i < matching_feature.size(); i++) {
			matching_point_before.push_back(all_keypoints[frame_num - frame_interval][matching_feature[i].queryIdx].pt);
			matching_point_after.push_back(all_keypoints[frame_num][matching_feature[i].trainIdx].pt);
		}
		printf("Gray  Matched feature points : %d\n", matching_feature.size());

	}

	return matching_feature.size();
}

int find_effective_matching_feature(Mat homography_mat, Optimize_parameter Optima_param, vector<DMatch> matching_feature,vector<DMatch> & matching_effective_feature,int matching_number, vector<vector<KeyPoint>> all_keypoints, vector<float> &disparity_x, int frame_num, int frame_interval, int color)
{

	color_name color_str;
	int count = 0;

	if (color == 0 || color == 1 || color == 2) {
		for (int match = 0; match < matching_number; match++) {
			double x_dist = abs(all_keypoints[3 * (frame_num - frame_interval) + color][matching_feature[match].queryIdx].pt.x - all_keypoints[3 * frame_num + color][matching_feature[match].trainIdx].pt.x);
			double y_dist = abs(all_keypoints[3 * (frame_num - frame_interval) + color][matching_feature[match].queryIdx].pt.y - all_keypoints[3 * frame_num + color][matching_feature[match].trainIdx].pt.y);
			double radian = atan2(y_dist, x_dist);	
			if (x_dist < Optima_param.distance_limit) {
				if (y_dist < Optima_param.distance_limit) {
					if (5 > radian * 180 / 3.145192)
					{
						matching_effective_feature.push_back(matching_feature[match]);

						float dis = all_keypoints[3 * (frame_num - frame_interval) + color][matching_feature[match].queryIdx].pt.x - all_keypoints[3 * frame_num + color][matching_feature[match].trainIdx].pt.x;
						disparity_x.push_back(dis);
						count++;
					}
				}
			}
		}
		printf("%s Effective matching feature : %d\n", color_str.color_str[color], count);

	}
	else
	{
		vector<float>radian;
		vector<float>x_dist;
		vector<float>y_dist;
		float radian_mean = 0;
		for (int match = 0; match < matching_number; match++) {
			float temp_x = abs(all_keypoints[frame_num - frame_interval][matching_feature[match].queryIdx].pt.x - all_keypoints[frame_num][matching_feature[match].trainIdx].pt.x);
			float temp_y = abs(all_keypoints[frame_num - frame_interval][matching_feature[match].queryIdx].pt.y - all_keypoints[frame_num][matching_feature[match].trainIdx].pt.y);
			if (temp_x < Optima_param.distance_limit && temp_y < Optima_param.distance_limit) {
				x_dist.push_back(temp_x);
				y_dist.push_back(temp_y);
				matching_effective_feature.push_back(matching_feature[match]);
				//disparity_x.push_back(all_keypoints[frame_num - frame_interval][matching_feature[match].queryIdx].pt.x - all_keypoints[frame_num][matching_feature[match].trainIdx].pt.x);
				disparity_x.push_back(temp_x);
				radian.push_back(180 * atan2(temp_y, temp_x) / 3.141592);
				count++;

			}
		}
		for (int i = 0; i < x_dist.size(); i++) {
			radian_mean += radian[i];
		}
		radian_mean /= x_dist.size();
		for (int i = 0; i < x_dist.size(); i++) {
			if (radian_mean * 1.3 < radian[i] && radian_mean * 0.7 > radian[i]) {
			
				//printf("radian : %f dist x %f, dist y %f\n", radian * 180 / 3.145192, x_dist, y_dist);

			}
		}


		printf("Gray  Effective matching feature : %d\n", count);

	}



	return count;
}


Min_Max refine_disparity_normalize_frame(Camera_parameter Camera_param, vector<DMatch>matching_effective_feature, vector<vector<KeyPoint>> all_keypoints, vector<float> disparity, vector<float>& depth, vector<float>& normalized_key_point_x, vector<float>& normalized_key_point_y, int cols, int rows, int effective_number, int frame_num, int frame_interval, int color)
{
	color_name color_str;
	vector<float> test;
	Min_Max min_max;
	for (int i = 0; i < effective_number; i++) {
		if (min_max.min > disparity[i])
			min_max.min = disparity[i];
		if (min_max.max < disparity[i])
			min_max.max = disparity[i];
	}
	if (color == 0 || color == 1 || color == 2) {

		//Refine Disparity
		for (int i = 0; i < effective_number; i++) {
			float buf = 0;
			//disparity[i] += 2 * abs(min_max.min);
			//disparity[i] = Camera_param.base_line * Camera_param.focal_length / disparity[i];
			buf = disparity[i] + 2 * abs(min_max.min);
			buf = (Camera_param.base_line * Camera_param.focal_length) / buf;
			depth.push_back(buf);
			//Normalize Frame
			if (cols < rows) {
				normalized_key_point_x.push_back(all_keypoints[3 * (frame_num - frame_interval) + color][matching_effective_feature[i].queryIdx].pt.x / cols);
				normalized_key_point_y.push_back(all_keypoints[3 * (frame_num - frame_interval) + color][matching_effective_feature[i].queryIdx].pt.y / cols);
			}
			else {
				normalized_key_point_x.push_back(all_keypoints[3 * (frame_num - frame_interval) + color][matching_effective_feature[i].queryIdx].pt.x / rows);
				normalized_key_point_y.push_back(all_keypoints[3 * (frame_num - frame_interval) + color][matching_effective_feature[i].queryIdx].pt.y / rows);
			}
		}
		printf("%s Min Max disparity : %.2f %.2f\n", color_str.color_str[color], min_max.min, min_max.max);
	}
	else
	{
		//cal min depth
		float min_depth = 0;
		for (int i = 0; i < effective_number; i++) {
			if (disparity[i] == min_max.max) {
				min_depth = disparity[i] + abs(min_max.min);
				min_depth = (Camera_param.base_line * Camera_param.focal_length) / min_depth;
			}
		}
		for (int i = 0; i < effective_number; i++) {
			if (disparity[i] != min_max.min && disparity[i] != min_max.max) {
				depth.push_back(disparity[i] / (min_max.max - min_max.min) * (min_depth - 1) + 1);
			}
			if (disparity[i] == min_max.min)
				depth.push_back(1);
			if (disparity[i] == min_max.max)
				depth.push_back(min_depth);
		}
		printf("Image Size :  cols %d rows %d ", cols, rows);
		//Refine Disparity
		for (int i = 0; i < effective_number; i++) {
			//Normalize Frame
			if (cols < rows) {
				normalized_key_point_x.push_back(all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.x / cols);
				normalized_key_point_y.push_back(all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.y / cols);
			}
			else {
				normalized_key_point_x.push_back(all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.x / rows);
				normalized_key_point_y.push_back(all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.y / rows);
			}
		}
		printf("Gray Min Max disparity : %.2f %.2f\n", min_max.min, min_max.max);
	}
	return min_max;
}



