#include "sub_module.h"
#include "util.h"

int main(int argc, char* argv[]) {


	//Varaiable declaration
	color_name color_str;
	string extenstion;
	string ori_frame_write_path = "./ori_frame/";
	string descriptor_write_path = "./descriptor/";
	char path[256];
	char frame_name[256];
	Camera_parameter Camera_param;
	Camera_param.focal_length = 0.012;
	Camera_param.base_line = 0;
	SIFT_parameter SIFT_param;
	SIFT_param.octave = 3;
	SIFT_param.feature_num = 0;
	SIFT_param.sigma = 1.6;
	SIFT_param.contrast = 0.04;
	SIFT_param.edge = 10;
	Matching_parameter Matching_param;
	Matching_param = { 0.5f };
	vector<float> middle_distance;
	Optimize_parameter Optima_param;

	Optima_param.distance_limit = 200;
	vector<vector<KeyPoint>> all_keypoints;
	vector<Mat> all_descriptors;
	vector<Mat> y;
	vector<Mat> rgb_frame;
	vector<vector<float>> all_depth;
	vector<int> all_effective_number;

	int frame_interval = 5;
	int total_frame = 0;
	FILE* log_file = fopen("./log.txt", "wt");
	mkdir("./descriptor/");
	mkdir("./matching/");
	mkdir("./excel/");
	mkdir("./point_cloud/");
	mkdir("./ori_frame/");
	mkdir("./keypoints/");

	FILE* fp_homography = fopen("./matching/homography.csv", "wt");
	extenstion = "./frames/0.mp4";
	if (extenstion.substr(extenstion.length() - 3, 3) == "mp4" || extenstion.substr(extenstion.length() - 3, 3) == "avi")
		total_frame = get_video_frame_gray(extenstion, y, rgb_frame);
	else
	{
		for (int i = 0; i < 91; i++)
		{
			sprintf_s(path, "./frames/%d.jpg", i, IMREAD_GRAYSCALE);
			Mat buf = imread(path);
			//cvtColor(buf, buf, COLOR_RGB2GRAY);
			y.push_back(buf);
		}
		total_frame = 91;
		frame_interval = 10;
	}
	int sum_j = 0;
	for (int i = 1; i <= total_frame; i++) {
		if (i % frame_interval == 0 && i != 0)
		{
			float temp = 0;
			for (int j = 0; j < frame_interval/3 * 5; j++) {
				temp += distance_ori[j + sum_j];
				//printf("%d %.12f\n", j + sum_j, distance_ori[j + sum_j]);
			}
			sum_j += 5;
			middle_distance.push_back(temp);
			printf("%.12f\n", temp);

		}
	}
	save_all_descriptor_gray(descriptor_write_path, SIFT_param, y, all_keypoints, all_descriptors, frame_interval);

	printf("\n\n\\\\\\\\\\\\\\\\Start Depth Estimation\\\\\\\\\\\\\n");
	printf("\n");
	float first_homography = 0;
	float disparity_optima = 0;
	float x_distance = 0;
	for (int frame_num = frame_interval+ frame_interval; frame_num < total_frame; frame_num += frame_interval) {
		int dis_count = 0;
		sprintf_s(path, "./excel/%d_match.csv", frame_num - frame_interval + 1);
		FILE* fp= fopen(path, "wt");
		printf("\n%d frame and %d frame estimate\n", frame_num + 1, frame_num + frame_interval + 1);

		//Matching feature points
		vector<Point2f> matching_point_before, matching_point_after;
		vector<DMatch> matching_feature;
		vector<DMatch> matching_effective_feature;
		vector<float>normalized_key_point_x, normalized_key_point_y;
		int matching_number;
		matching_number = matching_feature_points(Matching_param, all_keypoints, all_descriptors, matching_point_before, matching_point_after, matching_feature, frame_num, frame_interval, 4);

		//Find homography
		printf("\nGet Homography Matrix\n");
		Mat homography_mat;
		homography_mat = findHomography(matching_point_before, matching_point_after, RANSAC);
		if (frame_num == frame_interval + frame_interval) 
			first_homography = abs(homography_mat.at<double>(0, 2));
		else
			disparity_optima = abs(homography_mat.at<double>(0, 2)) / first_homography;
	
		Camera_param.base_line = middle_distance[dis_count+1] * disparity_optima;
		if (frame_num != frame_interval + frame_interval)
			x_distance += middle_distance[dis_count + 1];
		printf("Estimated BaseLine : %.5f \n\n", Camera_param.base_line);

		printf("Disparity Calculate and Find Effective Feature\n");
		printf("\n");
		vector<float> disparity;
		int effective_number;
		printf("Disparity Limit %d\n", Optima_param.distance_limit);
		effective_number = find_effective_matching_feature(homography_mat, Optima_param, matching_feature, matching_effective_feature, matching_number, all_keypoints, disparity, frame_num, frame_interval, 4);

		all_effective_number.push_back(effective_number);

		printf("\n");
		printf("Refine Disparity Image Coordinate Normalization\n");
		vector<float> depth;

		Min_Max dis;

		dis = refine_disparity_normalize_frame(Camera_param, matching_effective_feature, all_keypoints, disparity, depth, normalized_key_point_x, normalized_key_point_y, y[0].cols, y[0].rows, effective_number, frame_num, frame_interval, 4);
		printf("Gray Min MAX dis : %.2f %.2f\n", dis.min, dis.max);

	
		Min_Max depth_1;
		depth_1 = cal_min_max(depth);
		all_depth.push_back(depth);

		printf("\n");
		printf("Gray Min MAX Depth : %.2f %.2f\n", depth_1.min, depth_1.max);

		printf("\n");
		printf("Depth Normalizaion\n");

		float var_depth;
		float mean_depth;
		var_depth = variance(depth);
		mean_depth = mean(depth);
		printf("Var %f mean %f\n", var_depth, mean_depth);
	
		std::vector<float> red_color;
		std::vector<float> green_color;
		std::vector<float> blue_color;

		for (int i = 0; i < effective_number; i++) {
			red_color.push_back((unsigned char)rgb_frame[frame_num - frame_interval].at<Vec3b>((int)all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.y, (int)all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.x)[2]);
			green_color.push_back((unsigned char)rgb_frame[frame_num - frame_interval].at<Vec3b>((int)all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.y, (int)all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.x)[1]);
			blue_color.push_back((unsigned char)rgb_frame[frame_num - frame_interval].at<Vec3b>((int)all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.y, (int)all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.x)[0]);
		}
		struct dataType { Point3d point; int red; int green; int blue; };
		typedef dataType SpacePoint;
		vector<SpacePoint> pointCloud;

		sprintf_s(path, "./point_cloud/%d_pointcloud2.ply", frame_num - frame_interval+1);
		ofstream outfile2(path);
		outfile2 << "ply\n" << "format ascii 1.0\n" << "comment VTK generated PLY File\n";
		//outfile2 << "obj_info vtkPolyData points and polygons : vtk4.0\n" << "element vertex " << depth[0].size() + depth[1].size() + depth[2].size() << "\n";
		outfile2 << "obj_info vtkPolyData points and polygons : vtk4.0\n" << "element vertex " << depth.size() << "\n";
		outfile2 << "property float x\n" << "property float y\n" << "property float z\n";
		outfile2 << "property uchar red\n" << "property uchar green\n" << "property uchar blue\n";
		outfile2 << "end_header\n";
		for (int i = 0; i < depth.size(); i++) {
			//outfile2 << normalized_key_point_x[i]<< " ";
			outfile2 << normalized_key_point_x[i] + x_distance*15<< " ";
			outfile2 << -normalized_key_point_y[i] << " ";
			outfile2 << -depth[i] << " ";
			outfile2 << (int)red_color[i] << " ";
			outfile2 << (int)green_color[i] << " ";
			outfile2 << (int)blue_color[i] << " ";
			outfile2 << "\n";
		}
	
		fprintf(fp, "Matching points,,Effective,Disparity,Depth,base,%.5f,focal,%.5f\n", Camera_param.base_line, Camera_param.focal_length);
		for (int i = 0; i < matching_feature.size(); i++) {
			fprintf(fp, "%f,%f,", all_keypoints[frame_num - frame_interval][matching_feature[i].queryIdx].pt.x, all_keypoints[frame_num - frame_interval][matching_feature[i].queryIdx].pt.y);
			fprintf(fp, "%f,%f,", all_keypoints[frame_num][matching_feature[i].trainIdx].pt.x, all_keypoints[frame_num][matching_feature[i].trainIdx].pt.y);
			if (i < effective_number) {
				fprintf(fp, ",%f,%f,", all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.x, all_keypoints[frame_num - frame_interval][matching_effective_feature[i].queryIdx].pt.y);
				fprintf(fp, "%f,%f,", all_keypoints[frame_num][matching_effective_feature[i].trainIdx].pt.x, all_keypoints[frame_num][matching_effective_feature[i].trainIdx].pt.y);
			}
			if (i < depth.size()) {
				fprintf(fp, ",%f,%f,%f", disparity[i], disparity[i] + abs(dis.min), depth[i]);

			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		Mat img_matches;
		mkdir("./matching/");

		drawMatches(y[frame_num - frame_interval], all_keypoints[frame_num - frame_interval], y[frame_num], all_keypoints[frame_num], matching_effective_feature, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		sprintf_s(path, "./matching/gray.%d_%d_matching.bmp", frame_num - frame_interval+1, frame_num+1);
		imwrite(path, img_matches);
		cvtColor(y[frame_num - frame_interval], img_matches, COLOR_GRAY2BGR);
		for (int i = 0; i < matching_feature.size(); i++) {
			circle(img_matches
				, Point(all_keypoints[frame_num - frame_interval][matching_feature[i].queryIdx].pt.x
					, all_keypoints[frame_num - frame_interval][matching_feature[i].queryIdx].pt.y), 5, Scalar(0, 0, 255), -1, -1, 0);
		}
		mkdir("./matched_keypoint/");

		sprintf_s(path, "./matched_keypoint/%d.bmp", frame_num - frame_interval + 1);
		imwrite(path, img_matches);

		fprintf(fp_homography, "%d frame\n", frame_num - frame_interval+1);
		for (int i = 0; i < homography_mat.rows; i++) {
			for (int j = 0; j < homography_mat.cols; j++) {
				fprintf(fp_homography, "%f,", homography_mat.at<double>(i, j));
				printf("%f,", homography_mat.at<double>(i, j));
			}
			fprintf(fp_homography, "\n");
			printf("\n");
		}
		fprintf(fp_homography, "sx,%f,sy,%f,D,%f,P,%f\n\n\n"
			, sqrt(pow(homography_mat.at<double>(0, 0), 2) + pow(homography_mat.at<double>(1, 0), 2))
			, sqrt(pow(homography_mat.at<double>(0, 1), 2) + pow(homography_mat.at<double>(1, 1), 2))
			, homography_mat.at<double>(0, 0) * homography_mat.at<double>(1, 1) - homography_mat.at<double>(0, 1) * homography_mat.at<double>(1, 0)
			, sqrt(pow(homography_mat.at<double>(2, 0), 2) + pow(homography_mat.at<double>(2, 1), 2)));
		fprintf(fp_homography, "\n");

		if (frame_num + frame_interval > total_frame)
			break;

		dis_count++;
	}
	printf("\n\\\\\\\\\\\\\\\\\End Depth Estimation \\\\\\\\\\\ ");

	fclose(fp_homography);
	fclose(log_file);
}