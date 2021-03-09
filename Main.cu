#include "main.h"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


__global__ void checkVoxel(float* cameras, cuda::PtrStepSz<uchar3>* imgs, int* result, const int resolution, const int camCount, clock_t* time, int w, int h) {
	clock_t start = clock();
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < (resolution * resolution * resolution)) {
		float z = idx / (resolution * resolution);
		int zRemain = idx % (resolution * resolution);
		float y = resolution - zRemain / resolution;
		int x = idx % resolution;

		float posx = (x / resolution - 0.5) * 4;
		float posy = (y / resolution - 0.5) * 4;
		float posz = (z / resolution - 0.5) * 4;
		float posw = 1.0;
		int count = 0;
		float tt = 0.0f;
		for (int i = 0; i < camCount; i++) {

			float projectW = cameras[i * 16 + 3 * 4] * posx + cameras[i * 16 + 3 * 4 + 1] * posy + cameras[i * 16 + 3 * 4 + 2] * posz + cameras[i * 16 + 3 * 4 + 3] * posw;
			float projectX = (cameras[i * 16] * posx + cameras[i * 16 + 1] * posy + cameras[i * 16 + 2] * posz + cameras[i * 16 + 3] * posw) / projectW;
			float projectY = (cameras[i * 16 + 1 * 4] * posx + cameras[i * 16 + 1 * 4 + 1] * posy + cameras[i * 16 + 1 * 4 + 2] * posz + cameras[i * 16 + 1 * 4 + 3] * posw) / projectW;
			float projectZ = (cameras[i * 16 + 2 * 4] * posx + cameras[i * 16 + 2 * 4 + 1] * posy + cameras[i * 16 + 2 * 4 + 2] * posz + cameras[i * 16 + 2 * 4 + 3] * posw) / projectW;

			float u = projectY * 0.5 + 0.5;
			float v = projectX * 0.5 + 0.5;
			float point_z = projectZ * 0.5 + 0.5;

			if (u <= 1.0 && u >= 0.0 && v <= 1.0 && v >= 0.0) {
				u = 1 - u;
				float db = imgs[i](int(u* w), int(h* v)).x - 39;
				float dg = imgs[i](int(u* w), int(h* v)).y - 135;
				float dr = imgs[i](int(u* w), int(h* v)).z - 39;
				if ((sqrt(db * db + dg * dg + dr * dr) > 1))
					count += 1;
			}
		}
		if (count == camCount) result[idx] = 1;
		else result[idx] = 0;
	}
	else {
		//error
		result[idx] = 9;
	}
	*time = clock() - start;
}

cv::Mat MreadImage(const std::string& fileName)
{
	clock_t startTime = clock();
	cv::Mat mImg = cv::imread(fileName);
	clock_t endTime = clock();

	std::cout << "Time to read image: " << (float)(endTime - startTime) / (float)CLOCKS_PER_SEC << std::endl;
	return mImg;
}

int main(int argc, char** argv) {

	Json data = loadJson("camera.json");
	data["visualHull"] = "";

	for (int i = 0; i < 7; i++) {
		string fileName = visualHull(i, data["camera"]);
		cout << "before ," << data["visualHull"] << endl;
		string s = data["visualHull"];
		data["visualHull"] = s + fileName;
		cout << "after ," << data["visualHull"] << endl;
	}
	system("pause");
	//TODO
	writeJson();

	return 0;

}

Json loadJson(string fileName) {
	ifstream jsonFile(fileName);
	Json Json;
	jsonFile >> Json;

	//cout << "load Json" << endl;
	return Json;
};

string visualHull(int idx, Json cams) {
	//cout << "idx" << idx << endl;
	//params
	string imgFolder = "D:/lab/Visuallhull-cuda/StreamingAssets";
	int resolution = 128;
	int imgWidth, imgHeigh;
	vector<Mat> silhouetteImgs;

	int camCount = ceil(cams.size() / 3);

	//initial  

	float* camAry;

	camAry = (float*)malloc(camCount * 4 * 4 * sizeof(float));

	//img
	cuda::PtrStepSz<uchar3>* localImgs = (cuda::PtrStepSz<uchar3>*)malloc(camCount * sizeof(cuda::PtrStepSz<uchar3>));


	//read image & camera pos
	for (int i = 0; i < camCount; i++) {
		//cout << cams[i] << '\n';
		//cout << (double)cams[i]["world2screenMat"]["e00"] << endl;
		/*
		camParamsAry[i / 3][0][0] = (float)cams[i]["world2screenMat"]["e00"];
		camParamsAry[i / 3][0][1] = (float)cams[i]["world2screenMat"]["e01"];
		camParamsAry[i / 3][0][2] = (float)cams[i]["world2screenMat"]["e02"];
		camParamsAry[i / 3][0][3] = (float)cams[i]["world2screenMat"]["e03"];

		camParamsAry[i / 3][1][0] = (float)cams[i]["world2screenMat"]["e10"];
		camParamsAry[i / 3][1][1] = (float)cams[i]["world2screenMat"]["e11"];
		camParamsAry[i / 3][1][2] = (float)cams[i]["world2screenMat"]["e12"];
		camParamsAry[i / 3][1][3] = (float)cams[i]["world2screenMat"]["e13"];

		camParamsAry[i / 3][2][0] = (float)cams[i]["world2screenMat"]["e20"];
		camParamsAry[i / 3][2][1] = (float)cams[i]["world2screenMat"]["e21"];
		camParamsAry[i / 3][2][2] = (float)cams[i]["world2screenMat"]["e22"];
		camParamsAry[i / 3][2][3] = (float)cams[i]["world2screenMat"]["e23"];

		camParamsAry[i / 3][3][0] = (float)cams[i]["world2screenMat"]["e30"];
		camParamsAry[i / 3][3][1] = (float)cams[i]["world2screenMat"]["e31"];
		camParamsAry[i / 3][3][2] = (float)cams[i]["world2screenMat"]["e32"];
		camParamsAry[i / 3][3][3] = (float)cams[i]["world2screenMat"]["e33"];
		*/

		camAry[i * 16] = (float)cams[i * 3]["world2screenMat"]["e00"];
		camAry[i * 16 + 1] = (float)cams[i * 3]["world2screenMat"]["e01"];
		camAry[i * 16 + 2] = (float)cams[i * 3]["world2screenMat"]["e02"];
		camAry[i * 16 + 3] = (float)cams[i * 3]["world2screenMat"]["e03"];

		camAry[i * 16 + 1 * 4] = (float)cams[i * 3]["world2screenMat"]["e10"];
		camAry[i * 16 + 1 * 4 + 1] = (float)cams[i * 3]["world2screenMat"]["e11"];
		camAry[i * 16 + 1 * 4 + 2] = (float)cams[i * 3]["world2screenMat"]["e12"];
		camAry[i * 16 + 1 * 4 + 3] = (float)cams[i * 3]["world2screenMat"]["e13"];

		camAry[i * 16 + 2 * 4] = (float)cams[i * 3]["world2screenMat"]["e20"];
		camAry[i * 16 + 2 * 4 + 1] = (float)cams[i * 3]["world2screenMat"]["e21"];
		camAry[i * 16 + 2 * 4 + 2] = (float)cams[i * 3]["world2screenMat"]["e22"];
		camAry[i * 16 + 2 * 4 + 3] = (float)cams[i * 3]["world2screenMat"]["e23"];

		camAry[i * 16 + 3 * 4] = (float)cams[i * 3]["world2screenMat"]["e30"];
		camAry[i * 16 + 3 * 4 + 1] = (float)cams[i * 3]["world2screenMat"]["e31"];
		camAry[i * 16 + 3 * 4 + 2] = (float)cams[i * 3]["world2screenMat"]["e32"];
		camAry[i * 16 + 3 * 4 + 3] = (float)cams[i * 3]["world2screenMat"]["e33"];


		//image

		//remove first character
		string imgName = cams[i * 3]["img"][idx];
		imgName = imgName.substr(1, imgName.size() - 1);
		string path = imgFolder + imgName;

		cout << path << endl;
		Mat img;
		img = MreadImage(path);
		cout << "imread complete" << endl;
		if (img.empty()) {
			cout << "read null" << endl;
			system("pause");
		}
		cuda::GpuMat GPU_img;
		cv::Size s = img.size();
		imgWidth = s.width;
		imgHeigh = s.height;
		cout << "mat upload" << imgWidth << "," << imgHeigh << endl;
		//system("pause");
		GPU_img.upload(img);
		localImgs[i] = GPU_img;

		//cout << "mat upload complete" << endl;



	}

	/*
	vector<vector<vector<double>>> camParams;
	for (int i = 0; i < camCount; i++) {
		if (i % 3 == 0) {
			//cout << cams[i] << '\n';
			vector<vector<double>> curCamParams;

			vector<double> curCamParam0;
			curCamParam0.push_back(cams[i]["world2screenMat"]["e00"]);
			curCamParam0.push_back(cams[i]["world2screenMat"]["e01"]);
			curCamParam0.push_back(cams[i]["world2screenMat"]["e02"]);
			curCamParam0.push_back(cams[i]["world2screenMat"]["e03"]);
			vector<double> curCamParam1;
			curCamParam1.push_back(cams[i]["world2screenMat"]["e10"]);
			curCamParam1.push_back(cams[i]["world2screenMat"]["e11"]);
			curCamParam1.push_back(cams[i]["world2screenMat"]["e12"]);
			curCamParam1.push_back(cams[i]["world2screenMat"]["e13"]);
			vector<double> curCamParam2;
			curCamParam2.push_back(cams[i]["world2screenMat"]["e20"]);
			curCamParam2.push_back(cams[i]["world2screenMat"]["e21"]);
			curCamParam2.push_back(cams[i]["world2screenMat"]["e22"]);
			curCamParam2.push_back(cams[i]["world2screenMat"]["e23"]);
			vector<double> curCamParam3;
			curCamParam3.push_back(cams[i]["world2screenMat"]["e30"]);
			curCamParam3.push_back(cams[i]["world2screenMat"]["e31"]);
			curCamParam3.push_back(cams[i]["world2screenMat"]["e32"]);
			curCamParam3.push_back(cams[i]["world2screenMat"]["e33"]);

			curCamParams.push_back(curCamParam0);
			curCamParams.push_back(curCamParam1);
			curCamParams.push_back(curCamParam2);
			curCamParams.push_back(curCamParam3);

			camParams.push_back(curCamParams);
			//image

			//remove first character
			string imgName = cams[i]["img"][idx];

			imgName = imgName.substr(1, imgName.size() - 1);

			silhouetteImgs.push_back(imread(imgFolder + "/d" + imgName));

		}

	}
	*/
	cout << "read complete" << endl;
	//system("pause");

	cout << "resolution : " << resolution << endl;

	//copy data to device
	float* gpuCamParams;
	cudaMalloc(&gpuCamParams, camCount * 4 * 4 * sizeof(float));
	cudaCheckErrors("cuda malloc cam params");
	cudaMemcpy(gpuCamParams, camAry, camCount * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

	cuda::PtrStepSz<uchar3>* gpuImgs;
	cudaMalloc(&gpuImgs, camCount * sizeof(cuda::PtrStepSz<uchar3>));
	cudaCheckErrors("cuda malloc gpu img");
	cudaMemcpy(gpuImgs, localImgs, camCount * sizeof(cuda::PtrStepSz<uchar3>), cudaMemcpyHostToDevice);


	//alloc space for result(on GPU)
	int* gpuResult;
	cudaMalloc(&gpuResult, pow(resolution, 3) * sizeof(int));
	cudaCheckErrors("cuda malloc gpu result");
	int threads_per_block = 128;
	int blocks_per_grid = ceil(pow(resolution, 3) / threads_per_block);


	//calc
	clock_t* time;
	cudaMalloc(&time, sizeof(clock_t));
	checkVoxel << <blocks_per_grid, threads_per_block >> > (gpuCamParams, gpuImgs, gpuResult, resolution, camCount, time, imgWidth, imgHeigh);

	//sync
	cudaDeviceSynchronize();
	cudaCheckErrors("cuda sync");

	cout << "marching cube time : " << time << endl;

	//get result from gpu
	int* localResult;
	localResult = (int*)malloc(pow(resolution, 3) * sizeof(int));
	cout << "ready copy" << endl;
	cudaMemcpy(localResult, gpuResult, pow(resolution, 3) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cuda gpu result to cpu");
	//save to txt to validate
	fstream file;
	string fileName = "reader.txt";
	file.open(fileName, ios::out);
	int t = 16;
	for (int i = 0; i < pow(resolution, 3); i++) {
		if (i >= t * 2) {
			t *= 2;
			cout << "current " << t << ", total" << pow(resolution, 3) << endl;
		}
		file << localResult[i] << endl;
		//cout << localResult[i] << endl;
	}
	cout << "save txt complete" << endl;
	//TODO:
	//render result

	//result to obj

	//save result

	//free mem
	cout << "free cam" << endl;
	free(camAry);

	cout << "free gpu cam" << endl;
	cudaFree(gpuCamParams);

	cout << "free img" << endl;
	free(localImgs);
	cout << "free gpu img" << endl;
	cudaFree(gpuImgs);

	cout << "free res" << endl;
	free(localResult);

	return "";
}

void writeJson() {
	//TODO
}
