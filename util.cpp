#pragma once
#include "util.h"

Min_Max cal_min_max(vector<float> x) {
	Min_Max min_max;
	for (int i = 0; i < x.size(); i++) {
		if (min_max.min > x[i])
			min_max.min = x[i];
		if (min_max.max < x[i])
			min_max.max = x[i];
	}
	return min_max;
}

int normalize_vec(Min_Max min_max, vector<float> x, vector<float> &y, float ratio, float bias) {
	for (int i = 0; i < x.size(); i++) {
		float buf = 0;
		buf = x[i];
		buf -= min_max.min;
		buf /= (min_max.max - min_max.min);
		buf = buf * ratio;
		buf = buf + bias;
		y.push_back(buf);
	}
	return 0;
}

float variance(vector<float> x) {

	float mean_val = mean(x);
	float var_val = 0;
	for (int i = 0; i < x.size(); i++)
		var_val = var_val + pow(mean(x) - x[i], 2);

	return var_val/x.size();
}
float mean(vector<float> x) {
	
	float mean_val = 0;
	for (int i = 0; i < x.size(); i++)
		mean_val += x[i];
	return mean_val /= x.size();
}





