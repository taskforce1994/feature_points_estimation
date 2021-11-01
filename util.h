#pragma once
#include "sub_module.h"
Min_Max cal_min_max(vector<float> x);
int normalize_vec(Min_Max min_max, vector<float> x, vector<float>& y, float ratio, float bias);

float variance(vector<float> x);
float mean(vector<float> x);

