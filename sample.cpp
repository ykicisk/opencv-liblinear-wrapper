#include "cv_wrapper/LibLinearClassifier.h"
#include <opencv2/opencv.hpp>

const int MAX_DATA = 100;
int main(){
	cv_wrapper::LibLinearClassifier::Param p;
	p.bias = 1.0;
	auto cl = cv_wrapper::LibLinearClassifier::create(p);

	cv::Mat data(MAX_DATA, 2, CV_32FC1);
	std::vector<double> labels(MAX_DATA);

	cv::randu(data(cv::Rect(0, 0, 2, MAX_DATA/2)), cv::Scalar(0.0), cv::Scalar(1.0));
	cv::randu(data(cv::Rect(0, MAX_DATA/2, 2, MAX_DATA/2)), cv::Scalar(-1.0), cv::Scalar(0.0));

	for (int i = 0; i < MAX_DATA/2; i++) labels[i] = -1.0;
	for (int i = MAX_DATA/2; i < MAX_DATA; i++) labels[i] = 1.0;

	cl->train(data, labels);

	cv::Mat test1 = (cv::Mat_<float>(1, 2) << 1.0f, 1.0f);
	cv::Mat test2 = (cv::Mat_<float>(1, 2) << -0.3f, -0.4f);

	{
		double result = cl->predict(test1);
		std::cout << result << std::endl;
	}
	{
		double result = cl->predict(test2);
		std::cout << result << std::endl;
	}
	return 0;
}
