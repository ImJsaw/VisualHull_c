#include "main.h"

#define CHECK_ERROR(call){\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}

__global__ void checkVoxel(double* cameras, cuda::PtrStepSz<uchar3>* imgs, int* result, const int resolution, const int camCount, clock_t* time, int w, int h)
{
	clock_t start = clock();
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < (resolution * resolution * resolution)) {
		double z = idx / (resolution * resolution);
		int zRemain = idx % (resolution * resolution);
		double y = resolution - zRemain / resolution;
		int x = idx % resolution;

		double posx = (x / resolution - 0.5) * 4;
		double posy = (y / resolution - 0.5) * 4;
		double posz = (z / resolution - 0.5) * 4;
		double posw = 1.0;
		cameras[camCount] = 0;
		int count = 0;
		for (int i = 0; i < camCount; i++) {
			/*
			double projectW = cameras[i][3][0] * posx + cameras[i][3][1] * posy + cameras[i][3][2] * posz + cameras[i][3][3] * posw;
			double projectX = (cameras[i][0][0] * posx + cameras[i][0][1] * posy + cameras[i][0][2] * posz + cameras[i][0][3] * posw) / projectW;
			double projectY = (cameras[i][1][0] * posx + cameras[i][1][1] * posy + cameras[i][1][2] * posz + cameras[i][1][3] * posw) / projectW;
			double projectZ = (cameras[i][2][0] * posx + cameras[i][2][1] * posy + cameras[i][2][2] * posz + cameras[i][2][3] * posw) / projectW;
			*/

			double projectW = cameras[i * 16 + 3 * 4] * posx + cameras[i * 16 + 3 * 4 + 1] * posy + cameras[i * 16 + 3 * 4+2] * posz + cameras[i * 16 + 3 * 4+3] * posw;
			double projectX = (cameras[i*16] * posx + cameras[i*16+1] * posy + cameras[i*16+2] * posz + cameras[i*16+3] * posw) / projectW;
			double projectY = (cameras[i*16+1*4] * posx + cameras[i*16+1*4+1] * posy + cameras[i * 16 + 1*4+2] * posz + cameras[i * 16 + 1 * 4 + 3] * posw) / projectW;
			double projectZ = (cameras[i * 16 + 2 * 4] * posx + cameras[i * 16 + 2 * 4+1] * posy + cameras[i * 16 + 2 * 4+2] * posz + cameras[i * 16 + 2 * 4 + 3] * posw) / projectW;
			double u = projectY * 0.5 + 0.5;
			double v = projectX * 0.5 + 0.5;
			double point_z = projectZ * 0.5 + 0.5;

			if (u <= 1.0 && u >= 0.0 && v <= 1.0 && v >= 0.0){
				u = 1 - u;
				double db = imgs[i](int(u* w), int(h* v)).x - 39;
				double dg = imgs[i](int(u * w), int(h * v)).y - 135;
				double dr = imgs[i](int(u * w), int(h * v)).z - 39;
				if ((sqrt(db * db + dg * dg + dr * dr) > 1) && (point_z - (imgs[i](int(u * w), int(h * v)).x / 255) > 0))
					count += 1;
			}
		}

		if (count == camCount) result[idx] = 1;
		else result[idx] = 0;

	}
	*time = clock() - start;
}

cv::Mat MreadImage(std::ifstream& input)
{
	input.seekg(0, std::ios::end);
	size_t fileSize = input.tellg();
	input.seekg(0, std::ios::beg);

	if (fileSize == 0) {
		return cv::Mat();
	}

	std::vector<unsigned char> data(fileSize);
	input.read(reinterpret_cast<char*>(&data[0]), sizeof(unsigned char) * fileSize);

	if (!input) {
		return cv::Mat();
	}

	clock_t startTime = clock();
	cv::Mat mImg = cv::imdecode(cv::Mat(data), cv::IMREAD_COLOR);
	clock_t endTime = clock();
	std::cout << "Time to decode image: " << (float)(endTime - startTime) / (float)CLOCKS_PER_SEC << std::endl;

	return mImg;
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
	string path = "D:/lab/Visuallhull-cuda/resources/0_0.png";

	cv::Mat i1 = MreadImage(path);


	//std::ifstream imgStream(path.c_str(), std::ios::binary);
	//cv::Mat i2 = MreadImage(imgStream);
	/*
	Mat img = imread("test_0978_aligned.jpg", -1);  //修改成自己圖片路徑

	if (cv::gpu::getCudaEnabledDeviceCount() == 0)
		printf("NO CUDA\n");
	else
		printf("CUDA = %d\n", cv::gpu::getCudaEnabledDeviceCount());

	namedWindow("video demo", CV_WINDOW_AUTOSIZE);
	imshow("video demo", img);
	cvWaitKey(0);
	*/

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
	string imgFolder = "D:/lab/Visuallhull-cuda/resources";
	int resolution = 512;
	int imgWidth, imgHeigh;
	vector<Mat> silhouetteImgs;

	//initial  
	//camParms array (camcount * 4 * 4)
	double ***camParamsAry;
	int camCount = cams.size();
	camParamsAry = (double***)malloc(camCount/3 * sizeof(double **));
	for (int i = 0; i < camCount; i++) {
		camParamsAry[i] = (double**)malloc(4 * sizeof(double*));
		for (int j = 0; j < 4; j++) {
			camParamsAry[i][j] = (double*)malloc(4 * sizeof(double));
		}
	}
	
	//img
	cuda::PtrStepSz<uchar3>* localImgs = (cuda::PtrStepSz<uchar3>*)malloc(camCount / 3 * sizeof(cuda::PtrStepSz<uchar3>));
	

	//read image & camera pos
	for (int i = 0; i < camCount; i++) {
		if (i % 3 == 0) {
			//cout << cams[i] << '\n';
			//cout << (double)cams[i]["world2screenMat"]["e00"] << endl;
			camParamsAry[i / 3][0][0] = (double)cams[i]["world2screenMat"]["e00"];
			camParamsAry[i / 3][0][1] = (double)cams[i]["world2screenMat"]["e01"];
			camParamsAry[i / 3][0][2] = (double)cams[i]["world2screenMat"]["e02"];
			camParamsAry[i / 3][0][3] = (double)cams[i]["world2screenMat"]["e03"];

			camParamsAry[i / 3][1][0] = (double)cams[i]["world2screenMat"]["e10"];
			camParamsAry[i / 3][1][1] = (double)cams[i]["world2screenMat"]["e11"];
			camParamsAry[i / 3][1][2] = (double)cams[i]["world2screenMat"]["e12"];
			camParamsAry[i / 3][1][3] = (double)cams[i]["world2screenMat"]["e13"];

			camParamsAry[i / 3][2][0] = (double)cams[i]["world2screenMat"]["e20"];
			camParamsAry[i / 3][2][1] = (double)cams[i]["world2screenMat"]["e21"];
			camParamsAry[i / 3][2][2] = (double)cams[i]["world2screenMat"]["e22"];
			camParamsAry[i / 3][2][3] = (double)cams[i]["world2screenMat"]["e23"];

			camParamsAry[i / 3][3][0] = (double)cams[i]["world2screenMat"]["e30"];
			camParamsAry[i / 3][3][1] = (double)cams[i]["world2screenMat"]["e31"];
			camParamsAry[i / 3][3][2] = (double)cams[i]["world2screenMat"]["e32"];
			camParamsAry[i / 3][3][3] = (double)cams[i]["world2screenMat"]["e33"];
			//image

			//remove first character
			string imgName = cams[i]["img"][idx];
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
			//cout << "mat upload" << endl;
			//system("pause");
			GPU_img.upload(img);
			localImgs[i / 3] = GPU_img;

			//cout << "mat upload complete" << endl;
			//silhouetteImgs.push_back(img);

		}

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
	system("pause");

	cout << "resolution : " << resolution << endl;

	//copy data to device
	double* gpuCamParams;
	cudaMalloc(&gpuCamParams, camCount / 3 * 4 * 4 * sizeof(float));
	cudaMemcpy(gpuCamParams, camParamsAry, camCount / 3 * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

	cuda::PtrStepSz<uchar3>* gpuImgs;
	cudaMalloc(&gpuImgs, camCount / 3 * sizeof(cuda::PtrStepSz<uchar3>));
	cudaMemcpy(gpuImgs, localImgs, camCount / 3 * sizeof(cuda::PtrStepSz<uchar3>), cudaMemcpyHostToDevice);


	//alloc space for result(on GPU)
	int* gpuResult;
	cudaMalloc(&gpuResult, pow(resolution,3) * sizeof(int));
	int threads_per_block = 1024;
	int blocks_per_grid = ceil(pow(resolution, 3) / threads_per_block);


	//calc
	clock_t* time;
	cudaMalloc((void**)&time, sizeof(clock_t));
	checkVoxel <<<blocks_per_grid, threads_per_block >>> (gpuCamParams, gpuImgs, gpuResult, resolution, camCount, time, imgWidth, imgHeigh);
	//sync
	CHECK_ERROR(cudaDeviceSynchronize());

	cout << "marching cube time : " << time << endl;

	//get result from gpu
	int* localResult;
	localResult = (int*)malloc(pow(resolution,3) * sizeof(int));
	cudaMemcpy(&sum, localResult, pow(resolution, 3) * sizeof(int), cudaMemcpyDeviceToHost);
	
	//TODO:
	//render result

	//result to obj

	//save result

	//free mem
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			free(camParamsAry[i][j]);
		}
		free(camParamsAry[i]);
	}
	free(camParamsAry);
	cudaFree(gpuCamParams);

	free(localImgs);
	cudaFree(gpuImgs);

	free(localResult);


	return "";
}

void writeJson() {
	//TODO
}
