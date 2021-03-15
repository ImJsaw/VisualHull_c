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

//__global__ void checkVoxel(float* cameras, uchar3 * imgs, int* result, const int resolution, const int camCount, clock_t * time, int w, int h, uchar3 * imgs_v) {
__global__ void checkVoxel(float* cameras, uchar3 * imgs, int* result, const int resolution, const int camCount, clock_t * time, int w, int h) {
	clock_t start = clock();
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < (resolution * resolution * resolution)) {
		float z = idx / (resolution * resolution);
		float zRemain = float(idx % (resolution * resolution));
		float y = resolution - zRemain / resolution;
		float x = (float)(idx % resolution);

		float posx = (x / resolution - 0.5) * 4;
		float posy = (y / resolution - 0.5) * 4;
		float posz = (z / resolution - 0.5) * 4;
		float posw = 1.0;
		int count = 0;
		for (int i = 0; i < camCount; i++) {
			float projectW = cameras[i * 16 + 3 * 4] * posx + cameras[i * 16 + 3 * 4 + 1] * posy + cameras[i * 16 + 3 * 4 + 2] * posz + cameras[i * 16 + 3 * 4 + 3] * posw;
			float projectX = (cameras[i * 16]		  * posx + cameras[i * 16 + 1]		   * posy + cameras[i * 16 + 2]		   * posz + cameras[i * 16 + 3]			* posw) / projectW;
			float projectY = (cameras[i * 16 + 1 * 4] * posx + cameras[i * 16 + 1 * 4 + 1] * posy + cameras[i * 16 + 1 * 4 + 2] * posz + cameras[i * 16 + 1 * 4 + 3] * posw) / projectW;
			float projectZ = (cameras[i * 16 + 2 * 4] * posx + cameras[i * 16 + 2 * 4 + 1] * posy + cameras[i * 16 + 2 * 4 + 2] * posz + cameras[i * 16 + 2 * 4 + 3] * posw) / projectW;


			float u = projectY * 0.5 + 0.5;
			float v = projectX * 0.5 + 0.5;
			float point_z = projectZ * 0.5 + 0.5;

			if (u <= 1.0 && u >= 0.0 && v <= 1.0 && v >= 0.0) {
				u = 1 - u;
				/*
				uint db = (imgs[0](int(u* w), int(h* v)).x);
				uint dg = (imgs[0](int(u* w), int(h* v)).y);
				uint dr = (imgs[0](int(u* w), int(h* v)).z);
				*/
				int yIndex = u * w;
				int xIndex = h * v;
				//count = i * h * w + xIndex * h + yIndex;
				int db = imgs[i * h * w + yIndex * h + xIndex].x - 39;
				int dg = imgs[i * h * w + yIndex * h + xIndex].y - 135;
				int dr = imgs[i * h * w + yIndex * h + xIndex].z - 39;
				//imgs_v[i * h * w + xIndex * h + yIndex] = imgs[i * h * w + xIndex * h + yIndex];

				float pixelDiff = db * db + dg * dg + dr * dr;
				if (sqrt(pixelDiff) > 1 && point_z-(imgs[i * h * w + yIndex * h + xIndex].x / 255)>0) {
					count += 1;
				}
				/*
				*/
			}
		}
		if (count == camCount) result[idx] = 1;
		else result[idx] = 0;
		//result[idx] = count;
	}
	else {
		//error
		result[idx] = -1;
	}
	//check correct pass image
	/*
	if (idx < (camCount * w*h)) {
		imgs_v[idx] = imgs[idx];
	}
	*/
	*time = clock() - start;
}

cv::Mat MreadImage(const std::string& fileName)
{
	cv::Mat mImg = cv::imread(fileName);
	return mImg;
}

string imgFolder = "D:/lab/Visuallhull-cuda/StreamingAssets/";

int main(int argc, char** argv) {

	Json data = loadJson(imgFolder+"camera.json");
	data["visualHull"] = "";

	for (int i = 0; i < 1; i++) {
		string fileName = visualHull(i, data["camera"]);
		//cout << "before ," << data["visualHull"] << endl;
		string s = data["visualHull"];
		data["visualHull"] = s + fileName;
		//cout << "after ," << data["visualHull"] << endl;
	}
	//system("pause");
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
	int resolution = 512;
	int imgWidth = 1024, imgHeigh = 1024;

	int camCount = cams.size();
	//camCount = 2;

	//initial  
	clock_t start_time = clock();
	clock_t end_time;

	float* camAry;

	camAry = (float*)malloc(camCount * 4 * 4 * sizeof(float));

	//img
	cuda::PtrStepSz<uchar3>* localImgs = (cuda::PtrStepSz<uchar3>*)malloc(camCount * sizeof(cuda::PtrStepSz<uchar3>));
	uchar3* forceImg = (uchar3*)malloc(camCount * imgWidth * imgHeigh * sizeof(uchar3));

	//read image & camera pos
	for (int i = 0; i < camCount; i++) {

		camAry[i * 16] = (float)cams[i ]["world2screenMat"]["e00"];
		camAry[i * 16 + 1] = (float)cams[i]["world2screenMat"]["e01"];
		camAry[i * 16 + 2] = (float)cams[i]["world2screenMat"]["e02"];
		camAry[i * 16 + 3] = (float)cams[i]["world2screenMat"]["e03"];

		camAry[i * 16 + 1 * 4] = (float)cams[i ]["world2screenMat"]["e10"];
		camAry[i * 16 + 1 * 4 + 1] = (float)cams[i ]["world2screenMat"]["e11"];
		camAry[i * 16 + 1 * 4 + 2] = (float)cams[i ]["world2screenMat"]["e12"];
		camAry[i * 16 + 1 * 4 + 3] = (float)cams[i ]["world2screenMat"]["e13"];

		camAry[i * 16 + 2 * 4] = (float)cams[i ]["world2screenMat"]["e20"];
		camAry[i * 16 + 2 * 4 + 1] = (float)cams[i ]["world2screenMat"]["e21"];
		camAry[i * 16 + 2 * 4 + 2] = (float)cams[i ]["world2screenMat"]["e22"];
		camAry[i * 16 + 2 * 4 + 3] = (float)cams[i ]["world2screenMat"]["e23"];

		camAry[i * 16 + 3 * 4] = (float)cams[i ]["world2screenMat"]["e30"];
		camAry[i * 16 + 3 * 4 + 1] = (float)cams[i ]["world2screenMat"]["e31"];
		camAry[i * 16 + 3 * 4 + 2] = (float)cams[i ]["world2screenMat"]["e32"];
		camAry[i * 16 + 3 * 4 + 3] = (float)cams[i ]["world2screenMat"]["e33"];
		//image

		//remove first character
		string imgName = cams[i ]["img"][idx];
		imgName = imgName.substr(1, imgName.size() - 1);
		string path = imgFolder + imgName;

		//cout << path << endl;
		Mat img;
		img = MreadImage(path);
		//cout << "imread complete" << endl;
		if (img.empty()) {
			cout << "read null" << endl;
			system("pause");
		}
		/*
		imshow("test", img);
		waitKey(0);
		*/
		cv::Size s = img.size();
		imgWidth = s.width;
		imgHeigh = s.height;
		
		for (int w = 0; w < imgWidth; w++) {
			for (int h = 0; h < imgHeigh; h++) {
				uchar3 pixel;
				pixel.x = img.at<Vec3b>(w, h)[0];
				pixel.y = img.at<Vec3b>(w, h)[1];
				pixel.z = img.at<Vec3b>(w, h)[2];
				forceImg[i * imgWidth * imgHeigh + w * imgHeigh + h] = pixel;
			}
		}
		//cout << "pic size: width" << imgWidth << ", heigh" << imgHeigh << endl;
		//check pic 
		/*
		imshow("test", img);
		waitKey(0);

		Mat verify(s, CV_8UC3, Scalar(0));
		for (int w = 0; w < imgWidth; w++) {
			for (int h = 0; h < imgHeigh; h++) {
				uchar3 pixel = forceImg[i * imgWidth * imgHeigh + w * imgHeigh + h];
				verify.at<Vec3b>(w, h)[0] = pixel.x;
				verify.at<Vec3b>(w, h)[1] = pixel.y;
				verify.at<Vec3b>(w, h)[2] = pixel.z;
			}
		}

		imshow("test", verify);
		waitKey(0);
		*/


	}

	end_time = clock();
	cout << "read img + json time : " << (float)(end_time - start_time) / (float)CLOCKS_PER_SEC << "s" << endl;

	cout << "resolution : " << resolution << endl;


	start_time = clock();
	//copy data to device
	float* gpuCamParams;
	cudaMalloc(&gpuCamParams, camCount * 4 * 4 * sizeof(float));
	cudaCheckErrors("cuda malloc cam params");
	cudaMemcpy(gpuCamParams, camAry, camCount * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

	uchar3* gpuImgs;
	cudaMalloc(&gpuImgs, camCount * imgWidth * imgHeigh * sizeof(uchar3));
	cudaCheckErrors("cuda malloc gpu img");
	cudaMemcpy(gpuImgs, forceImg, camCount * imgHeigh * imgWidth * sizeof(uchar3), cudaMemcpyHostToDevice);

	//allocate test space
	/*
	uchar3* gpuImgs_v;
	cudaMalloc(&gpuImgs_v, camCount* imgWidth* imgHeigh * sizeof(uchar3));
	*/

	//alloc space for result(on GPU)
	int* gpuResult;
	cudaMalloc(&gpuResult, pow(resolution, 3) * sizeof(int));
	cudaCheckErrors("cuda malloc gpu result");
	int threads_per_block = 128;
	int blocks_per_grid = ceil(pow(resolution, 3) / threads_per_block);
	end_time = clock();
	cout << "cpu to gpu mem time : " << (float)(end_time - start_time) / (float)CLOCKS_PER_SEC << "s" << endl;

	//calc
	clock_t* time;
	cudaMalloc(&time, sizeof(clock_t));

	start_time = clock();
	checkVoxel << <blocks_per_grid, threads_per_block >> > (gpuCamParams, gpuImgs, gpuResult, resolution, camCount, time, imgWidth, imgHeigh);
//	checkVoxel << <blocks_per_grid, threads_per_block >> > (gpuCamParams, gpuImgs, gpuResult, resolution, camCount, time, imgWidth, imgHeigh, gpuImgs_v);

	//cout << "calc" << endl;
	//system("pause");
	//sync
	cudaDeviceSynchronize();
	cudaCheckErrors("cuda sync");

	end_time = clock();
	cout << "marching cube time : " << (float)(end_time-start_time)/ (float)CLOCKS_PER_SEC << "s" << endl;
	//cudaMemcpy(&time_used, time, sizeof(clock_t),cudaMemcpyDeviceToHost);
	//system("pause");
	/*
	uchar3* verifyImg = (uchar3*)malloc(camCount * imgWidth * imgHeigh * sizeof(uchar3));
	cudaMemcpy(verifyImg, gpuImgs_v, camCount* imgHeigh* imgWidth * sizeof(uchar3), cudaMemcpyDeviceToHost);
	for (int i = 0; i < camCount; i++) {

		Mat verify(Size(imgWidth,imgHeigh), CV_8UC3, Scalar(0));
		for (int w = 0; w < imgWidth; w++) {
			for (int h = 0; h < imgHeigh; h++) {
				uchar3 pixel = verifyImg[i * imgWidth * imgHeigh + w * imgHeigh + h];
				verify.at<Vec3b>(w, h)[0] = pixel.x;
				verify.at<Vec3b>(w, h)[1] = pixel.y;
				verify.at<Vec3b>(w, h)[2] = pixel.z;
			}
		}
		imshow("test", verify);
		waitKey(0);
	}
	*/

	//get result from gpu
	start_time = clock();
	int* localResult;
	localResult = (int*)malloc(pow(resolution, 3) * sizeof(int));
	cout << "ready copy" << endl;
	cudaMemcpy(localResult, gpuResult, pow(resolution, 3) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cuda gpu result to cpu");
	cudaDeviceSynchronize();
	cudaCheckErrors("cuda sync");
	end_time = clock();
	cout << "gpu to cpu mem time : " << (float)(end_time - start_time) / (float)CLOCKS_PER_SEC << "s" << endl;
	//save to txt to validate
	
	fstream file;
	string fileName = "reader_256.txt";
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
	
	//cout << "save txt complete" << endl;
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
