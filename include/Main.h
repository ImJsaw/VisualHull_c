#pragma once

#include <iostream>
#include <string>

#include <json.hpp>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;
using namespace std;

using Json = nlohmann::json;

Json loadJson(string);
string visualHull(int idx, Json cams);
void writeJson();
