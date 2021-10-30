#pragma once

//OpenCV Header
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
//Standard Header
#include <functional>
#include <vector>
#include <queue>
using namespace std;
using std::cout;
using std::endl;

struct Camera_parameter {
	float focal_length = 0.012;
	float base_line = 0;
};
struct SIFT_parameter {
	int octave = 5;
	double contrast = 0.4;
	double edge = 40;
	double sigma = 0.4;
	int feature_num = 100000;
};
struct Optimize_parameter {
	int distance_limit = 200;
};
struct Matching_parameter {
	float ratio_thresh = 2.0f;
};
struct Point_info {
	vector<Point3f> duplicate_point;
	vector<Point3f> new_point;
	vector<Point3f> save_point;
	vector<Point3f> except_point;
};
struct Min_Max {
	float min = LONG_MAX;
	float max = LONG_MIN;
};

struct color_name {
	string color_str[3] = { "Red", "Green", "Blue" };
};

int get_video_frame(string path, vector<Mat>& rgb_frame);
int get_video_frame_gray(string path, vector<Mat>& gray, vector<Mat>& rgb_frame);
int rgb_split(string write_path, vector<Mat>& r, vector<Mat>& g, vector<Mat>& b, vector<Mat> rgb_frame);
int save_all_descriptor(string write_path, SIFT_parameter SIFT_param, vector<Mat> r, vector<Mat> g, vector<Mat> b, vector<vector<KeyPoint>>& all_keypoints, vector<Mat>& all_descriptors);
int save_all_descriptor_gray(string write_path, SIFT_parameter SIFT_param, vector<Mat> y, vector<vector<KeyPoint>>& all_keypoints, vector<Mat>& all_descriptors, int frame_interval);
int matching_feature_points(Matching_parameter Matching_param, vector<vector<KeyPoint>> all_keypoints, vector<Mat>& all_descriptors, vector<Point2f>& matching_point_before, vector<Point2f>& matching_point_after, vector<DMatch>& matching_feature, int frame_num, int frame_interval, int color);

int find_effective_matching_feature(Mat homography_mat,Optimize_parameter Optima_param, vector<DMatch> matching_feature, vector<DMatch>& matching_effective_feature, int matching_number, vector<vector<KeyPoint>> all_keypoints, vector<float>& disparity, int frame_num, int frame_interval, int color);

Min_Max refine_disparity_normalize_frame(Camera_parameter Camera_param, vector<DMatch>matching_effective_feature, vector<vector<KeyPoint>> all_keypoints, vector<float> disparity_x, vector<float>& depth, vector<float>& normalized_key_point_x, vector<float>& normalized_key_point_y, int cols, int rows, int effective_number, int frame_num, int frame_interval, int color);


double distance_ori[371] = {
	0.000404256	,
0.000800662	,
0.000800662	,
0.000800633	,
0.000797671	,
0.000794054	,
0.000802406	,
0.000803208	,
0.000786457	,
0.000777399	,
0.000781123	,
0.000805792	,
0.00083303	,
0.000826389	,
0.00080031	,
0.000786886	,
0.000781745	,
0.000790282	,
0.000809867	,
0.00081202	,
0.000794846	,
0.000782153	,
0.000792733	,
0.000843245	,
0.000881364	,
0.000822825	,
0.000735273	,
0.000726732	,
0.000763074	,
0.000779968	,
0.000786488	,
0.000800385	,
0.000814449	,
0.000825692	,
0.000827772	,
0.000803064	,
0.00077176	,
0.000758274	,
0.000764256	,
0.000802193	,
0.000840843	,
0.000833867	,
0.00079326	,
0.000758405	,
0.000757916	,
0.000806729	,
0.000849366	,
0.000815173	,
0.000764092	,
0.000785553	,
0.000840517	,
0.000840168	,
0.000804696	,
0.000780324	,
0.000763273	,
0.000740979	,
0.000736539	,
0.000777119	,
0.000829067	,
0.000856518	,
0.000854089	,
0.000803737	,
0.000761875	,
0.000796237	,
0.000844999	,
0.000843585	,
0.00081772	,
0.00078412	,
0.000753361	,
0.000736179	,
0.000720118	,
0.00071994	,
0.000741665	,
0.000806725	,
0.000868976	,
0.000862134	,
0.000835183	,
0.000809247	,
0.000790924	,
0.00079893	,
0.000804005	,
0.000806272	,
0.000820248	,
0.000805357	,
0.00078906	,
0.00082613	,
0.000855369	,
0.000855708	,
0.000850557	,
0.000818487	,
0.000786047	,
0.000775024	,
0.000775624	,
0.000797213	,
0.000806368	,
0.000790503	,
0.000773408	,
0.000788453	,
0.000820578	,
0.000819168	,
0.000793541	,
0.000747399	,
0.000720557	,
0.000765003	,
0.000816311	,
0.00081791	,
0.00078366	,
0.000746056	,
0.000741302	,
0.000769414	,
0.000795206	,
0.000798969	,
0.000788248	,
0.000782529	,
0.000791261	,
0.000813806	,
0.000841448	,
0.000836496	,
0.00080507	,
0.00077911	,
0.000778171	,
0.000820621	,
0.000865549	,
0.000847883	,
0.000802415	,
0.000784841	,
0.000810213	,
0.00084992	,
0.000852005	,
0.000808316	,
0.000774928	,
0.000785703	,
0.000819349	,
0.000874432	,
0.000905423	,
0.000850092	,
0.000793661	,
0.000802188	,
0.000840511	,
0.000859387	,
0.000831176	,
0.000795371	,
0.000796524	,
0.000835095	,
0.000859521	,
0.00084507	,
0.000831455	,
0.000843674	,
0.000846566	,
0.000825916	,
0.000811427	,
0.00081334	,
0.000811193	,
0.000809431	,
0.000827073	,
0.00085046	,
0.000847981	,
0.000812229	,
0.000779591	,
0.00080778	,
0.000878443	,
0.00092007	,
0.000883591	,
0.000793907	,
0.000745932	,
0.000788106	,
0.000844286	,
0.000829957	,
0.000789368	,
0.000775934	,
0.000789634	,
0.000822974	,
0.000846796	,
0.000849389	,
0.000837834	,
0.000843101	,
0.000847353	,
0.000827443	,
0.000829015	,
0.000831339	,
0.000811982	,
0.000811076	,
0.000837338	,
0.000891477	,
0.000927473	,
0.000889336	,
0.000847556	,
0.000848226	,
0.000857288	,
0.000854168	,
0.000851615	,
0.000847522	,
0.000824594	,
0.000794943	,
0.000792553	,
0.000844693	,
0.000891424	,
0.000882624	,
0.000853432	,
0.000835937	,
0.000847087	,
0.00085937	,
0.000849385	,
0.000837338	,
0.000831641	,
0.000803085	,
0.00077436	,
0.000798399
};






