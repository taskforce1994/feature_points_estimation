#include "../sub_module.h"
#include "../util.h"
#include "../example.hpp"


int main(int argc, char* argv[]) {
	//Varaiable declaration
	color_name color_str;
	string extenstion;
	string ori_frame_write_path = "./ori_frame/";
	string descriptor_write_path = "./descriptor/";
	char path[256];
	char frame_name[256];
	Camera_parameter Camera_param;
	Camera_param.focal_length = 12;
	Camera_param.base_line = 0;
	SIFT_parameter SIFT_param;
	SIFT_param.octave = 3;
	SIFT_param.feature_num = 0;
	SIFT_param.sigma = 1.6;
	SIFT_param.contrast = 0.08;
	SIFT_param.edge = 40;
	Matching_parameter Matching_param;
	Matching_param = { 1.0f };

	Optimize_parameter Optima_param;

	Optima_param = { 0.0001 , 0, 200, 100, 0 };

	vector<vector<KeyPoint>> all_keypoints;
	vector<Mat> all_descriptors;
	vector<Mat> r, g, b, rgb_frame;

	int frame_interval = 5;

	FILE* log_file = fopen("./log.txt", "wt");
	FILE* fp_homography = fopen("./matching/homography.csv", "wt");
	extenstion = "./frames/0.mp4";
	if (extenstion.substr(extenstion.length() - 3, 3) == "mp4" || extenstion.substr(extenstion.length() - 3, 3) == "avi")
		get_video_frame(extenstion, rgb_frame);


	rgb_split(ori_frame_write_path, r, g, b, rgb_frame);

	save_all_descriptor(descriptor_write_path, SIFT_param, r, g, b, all_keypoints, all_descriptors);

	printf("\n\n\\\\\\\\\\\\\\\\Start Depth Estimation\\\\\\\\\\\\\n");
	printf("\n");
	for (int frame_num = frame_interval; frame_num < 11; frame_num += frame_interval) {

		if (frame_num + frame_interval > 11)
			break;
		printf("\n%d frame and %d frame estimate\n", frame_num + 1, frame_num + frame_interval + 1);

		//Matching feature points
		vector<Point2f> matching_point_before[3], matching_point_after[3];
		vector<DMatch> matching_feature[3];
		vector<DMatch> matching_effective_feature[3];
		vector<float>normalized_key_point_x[3], normalized_key_point_y[3];
		int matching_number[3];
		matching_number[0]= matching_feature_points(Matching_param, all_keypoints, all_descriptors, matching_point_before[0], matching_point_after[0], matching_feature[0], frame_num, frame_interval, 0);
		matching_number[1] = matching_feature_points(Matching_param, all_keypoints, all_descriptors, matching_point_before[1], matching_point_after[1], matching_feature[1], frame_num, frame_interval, 1);
		matching_number[2] = matching_feature_points(Matching_param, all_keypoints, all_descriptors, matching_point_before[2], matching_point_after[2], matching_feature[2], frame_num, frame_interval, 2);


		//Find homography
		printf("\nGet Homography Matrix\n");
		Mat homography_mat[3];
		homography_mat[0] = findHomography(matching_point_before[0], matching_point_after[0], RANSAC);
		homography_mat[1] = findHomography(matching_point_before[1], matching_point_after[1], RANSAC);
		homography_mat[2] = findHomography(matching_point_before[2], matching_point_after[2], RANSAC);
		printf("RGB X axis translation : %.2f %.2f %.2f\n", homography_mat[0].at<double>(0, 2), homography_mat[1].at<double>(0, 2), homography_mat[2].at<double>(0, 2));

		//Camera baseline setting
		Camera_param.base_line = abs(homography_mat[0].at<double>(0, 2) + homography_mat[1].at<double>(0, 2) + homography_mat[2].at<double>(0, 2)) / 3;
		printf("Estimated BaseLine : %.2f \n\n", Camera_param.base_line);


		printf("Disparity Calculate and Find Effective Feature\n");
		printf("\n");
		vector<float> disparity_x[3];
		int effective_number[3];
		effective_number[0] = find_effective_matching_feature(Optima_param, matching_feature[0], matching_effective_feature[0], matching_number[0], all_keypoints, disparity_x[0], frame_num, frame_interval, 0);
		effective_number[1] = find_effective_matching_feature(Optima_param, matching_feature[1], matching_effective_feature[1], matching_number[1], all_keypoints, disparity_x[1], frame_num, frame_interval, 1);
		effective_number[2] = find_effective_matching_feature(Optima_param, matching_feature[2], matching_effective_feature[2], matching_number[2], all_keypoints, disparity_x[2], frame_num, frame_interval, 2);


		float min_dis[3] = { 0, }, max_dis[3] = { 0, };
		printf("\n");
		printf("Refine Disparity Image Coordinate Normalization\n");

		Min_Max dis[3];
		dis[0] = refine_disparity_normalize_frame(Camera_param, matching_effective_feature[0], all_keypoints, disparity_x[0], normalized_key_point_x[0], normalized_key_point_y[0], rgb_frame[0].cols, rgb_frame[0].rows, effective_number[0], frame_num, frame_interval, 0);
		dis[1] = refine_disparity_normalize_frame(Camera_param, matching_effective_feature[1], all_keypoints, disparity_x[1], normalized_key_point_x[1], normalized_key_point_y[1], rgb_frame[0].cols, rgb_frame[0].rows, effective_number[1], frame_num, frame_interval, 1);
		dis[2] = refine_disparity_normalize_frame(Camera_param, matching_effective_feature[2], all_keypoints, disparity_x[2], normalized_key_point_x[2], normalized_key_point_y[2], rgb_frame[0].cols, rgb_frame[0].rows, effective_number[2], frame_num, frame_interval, 2);

		Min_Max depth_1[3];

		//	float third_min_x = LONG_MAX;
		//	float third_max_x = LONG_MIN;

		//	//-- Average Depth 
		//	for (int i = 0; i < count_x; i++) {
		//		real_key_point_val[i] -= secon_min_x;
		//		real_key_point_val[i] += 0.01;
		//		real_key_point_val[i] /= (secon_max_x - secon_min_x);
		//		if (third_min_x > real_key_point_val[i])
		//			third_min_x = real_key_point_val[i];
		//		if (third_max_x < real_key_point_val[i])
		//			third_max_x = real_key_point_val[i];
		//	}
		//
		//
		//	std::vector<float> real_key_point_depth; std::vector<float> real_key_point_depth_x; std::vector<float> real_key_point_depth_y;
		//	float aver_dep[100] = { 0, };
		//	int aver_count[100] = { 0, };
		//	int isnan_test = 0;
		//
		//
		//	for (int m = 0; m < 10; m++)
		//		for (int n = 0; n < 10; n++)
		//			for (int i = 0; i < count_x; i++)
		//				if ((m * 0.1) < real_key_point_x[i] && real_key_point_x[i] < 0.1 + (m * 0.1) && (n * 0.1) < real_key_point_y[i] && real_key_point_y[i] < 0.1 + (n * 0.1)) {
		//					aver_dep[n * 10 + m] += real_key_point_val[i];
		//					aver_count[n * 10 + m]++;
		//				}
		//
		//
		//	float average_depth = 0;
		//
		//	for (int m = 0; m < 10; m++)
		//		for (int n = 0; n < 10; n++) {
		//			printf("%d %d (m,n) %f\t", m, n, aver_dep[n * 10 + m]);
		//			aver_dep[n * 10 + m] /= aver_count[n * 10 + m];
		//			printf("%f\t", aver_dep[n * 10 + m]);
		//
		//			if (std::isnan(aver_dep[n * 10 + m])) {
		//				aver_dep[n * 10 + m] = 0;
		//				isnan_test++;
		//			}
		//			average_depth += aver_dep[n * 10 + m];
		//			printf("%f is nan %d\n", aver_dep[n * 10 + m], isnan_test);
		//
		//		}
		//	average_depth /= 100 - isnan_test;
		//
		//	for (int i = 0; i < count_x; i++) {
		//		for (int m = 0; m < 10; m++)
		//			for (int n = 0; n < 10; n++)
		//				if ((m * 0.1) < real_key_point_x[i] && real_key_point_x[i] < 0.1 + (m * 0.1) && (n * 0.1) < real_key_point_y[i] && real_key_point_y[i] < 0.1 + (n * 0.1)) {
		//					real_key_point_val[i] = aver_dep[n * 10 + m];
		//					real_key_point_depth.push_back(aver_dep[n * 10 + m]);
		//				}
		//		real_key_point_depth_x.push_back(real_key_point_x[i]);
		//		real_key_point_depth_y.push_back(real_key_point_y[i]);
		//		Optima_param.clip_count++;
		//
		//	}
		//
		//	Optima_param.average_depth = average_depth;
		//	//for (int i = 0; i < count_x; i++) {
		//	//	//if (real_key_point_val[i] < average_depth + Optima_param.clip * (average_depth / 100) && real_key_point_val[i] > average_depth - Optima_param.clip * (average_depth / 100)) {
		//	//	
		//	//		printf("test %f %f %f\n", average_depth, real_key_point_val[i], real_key_point_depth[i]);
		//	//	//}
		//	//}
		//	float _min_x = LONG_MAX;
		//	float _max_x = LONG_MIN;
		//	for (int i = 0; i < Optima_param.clip_count; i++) {
		//		if (_min_x > real_key_point_depth[i]) {
		//			_min_x = real_key_point_depth[i];
		//		}
		//		if (_max_x < real_key_point_depth[i]) {
		//			_max_x = real_key_point_depth[i];
		//		}
		//	}
		//	printf("average_depth %f %f %f\n", average_depth, _min_x, _max_x);
		//	for (int i = 0; i < Optima_param.clip_count; i++) {
		//		real_key_point_depth[i] -= _min_x;
		//		real_key_point_depth[i] /= (_max_x - _min_x);
		//	}
		//
		//	struct dataType { Point3d point; int red; int green; int blue; };
		//	typedef dataType SpacePoint;
		//	vector<SpacePoint> pointCloud;
		//	sprintf_s(path, "./point_cloud/%d_pointcloud2.ply", frame_num - frame_interval);
		//	ofstream outfile2(path);
		//	outfile2 << "ply\n" << "format ascii 1.0\n" << "comment VTK generated PLY File\n";
		//	outfile2 << "obj_info vtkPolyData points and polygons : vtk4.0\n" << "element vertex " << Optima_param.clip_count << "\n";
		//	outfile2 << "property float x\n" << "property float y\n" << "property float z\n";
		//	outfile2 << "end_header\n";
		//	for (int i = 0; i < Optima_param.clip_count; i++) {
		//		outfile2 << real_key_point_depth_x[i] << " ";
		//		outfile2 << -real_key_point_depth_y[i] << " ";
		//		outfile2 << -real_key_point_depth[i] << " ";
		//		outfile2 << "\n";
		//	}
		//
		//
		//	printf("Parameter Check\n");
		//	printf("*******************************************************************************\n\n");
		//	printf("%d Frame \n", frame_num);
		//	printf("Focal length : %.2f BaseLine : %.2f                                           \n", Camera_param.focal_length, Camera_param.base_line);
		//	printf("SIFT : Octave : %d Contrast %.2f Edge %.0f Sigma %.2f Number of feature : %d \n"
		//		, SIFT_param.octave, SIFT_param.contrast, SIFT_param.edge
		//		, SIFT_param.sigma, SIFT_param.feature_num);
		//	printf("Optimization Min Disparity  : %.2f Average Depth : %.2f  Distance Limit : %d \n", Optima_param.min_disparity, Optima_param.average_depth, Optima_param.distance_limit);
		//	printf("Valid Points Number : %d\n", count_x);
		//	printf("*******************************************************************************\n\n");
		//
		//	fprintf(log_file, "Parameter Check\n");
		//	fprintf(log_file, "*******************************************************************************\n\n");
		//	fprintf(log_file, "%d Frame \n", frame_num);
		//	fprintf(log_file, "Focal length : %.2f BaseLine : %.2f                                           \n", Camera_param.focal_length, Camera_param.base_line);
		//	fprintf(log_file, "SIFT : Octave : %d Contrast %.2f Edge %.0f Sigma %.2f Number of feature : %d \n"
		//		, SIFT_param.octave, SIFT_param.contrast, SIFT_param.edge
		//		, SIFT_param.sigma, SIFT_param.feature_num);
		//	fprintf(log_file, "Optimization Min Disparity  : %.2f Average Depth : %.2f  Distance Limit : %d \n", Optima_param.min_disparity, Optima_param.average_depth, Optima_param.distance_limit);
		//	fprintf(log_file, "Valid Points Number : %d\n", count_x);
		//	fprintf(log_file, "*******************************************************************************\n\n");
		//
		//	//Data Save
		//	sprintf_s(path, "./descriptor/%d_frame.bmp", frame_num);
		//	imwrite(path, all_descriptors[frame_num]);
		//	//sprintf_s(path, "./frames/%d_frame.bmp", frame_num);
		//	//imwrite(path, rgb_frame[frame_num]);
		//	Mat img_matches;
		//	drawMatches(rgb_frame[frame_num - frame_interval], all_keypoints[frame_num - frame_interval], rgb_frame[frame_num], all_keypoints[frame_num], matching_effective_feature, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//	sprintf_s(path, "./matching/%d_%d_matching.bmp", frame_num - frame_interval, frame_num);
		//	imwrite(path, img_matches);
		//	drawMatches(rgb_frame[frame_num - frame_interval], all_keypoints[frame_num - frame_interval], rgb_frame[frame_num], all_keypoints[frame_num], matching_feature, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//	sprintf_s(path, "./matching/%d_%d_matching_ori.bmp", frame_num - frame_interval, frame_num);
		//	imwrite(path, img_matches);
		//	fprintf(fp_homography, "%d frame\n", frame_num - frame_interval);
		//	for (int i = 0; i < homography_mat.rows; i++) {
		//		for (int j = 0; j < homography_mat.cols; j++) {
		//			fprintf(fp_homography, "%f,", homography_mat.at<double>(i, j));
		//			printf("%f,", homography_mat.at<double>(i, j));
		//		}
		//		fprintf(fp_homography, "\n");
		//		printf("\n");
		//	}
		//	fprintf(fp_homography, "sx,%f,sy,%f,D,%f,P,%f\n\n\n"
		//		, sqrt(pow(homography_mat.at<double>(0, 0), 2) + pow(homography_mat.at<double>(1, 0), 2))
		//		, sqrt(pow(homography_mat.at<double>(0, 1), 2) + pow(homography_mat.at<double>(1, 1), 2))
		//		, homography_mat.at<double>(0, 0) * homography_mat.at<double>(1, 1) - homography_mat.at<double>(0, 1) * homography_mat.at<double>(1, 0)
		//		, sqrt(pow(homography_mat.at<double>(2, 0), 2) + pow(homography_mat.at<double>(2, 1), 2)));
		//	fprintf(fp_homography, "\n");
	}

	printf("\n\\\\\\\\\\\\\\\\\End Depth Estimation \\\\\\\\\\\ ");

	fclose(fp_homography);
	fclose(log_file);
}