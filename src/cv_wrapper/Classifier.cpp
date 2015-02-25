#include "cv_wrapper/Classifier.h"

namespace cv_wrapper{
	void Classifier::reset_scale(){
		norm_coef = cv::Mat();
		norm_bias = cv::Mat();
	}

	void Classifier::normalize(cv::InputArray src, cv::OutputArray dst){
		if (norm_coef.empty() || norm_bias.empty()){
			cv::Mat srcMat = src.getMat();
			srcMat.copyTo(dst);
		} else{
			cv::Mat srcMat = src.getMat();

			cv::Mat coef = cv::repeat(norm_coef, srcMat.rows, 1);
			cv::Mat bias = cv::repeat(norm_bias, srcMat.rows, 1);

			cv::Mat dstMat = coef.mul(srcMat) + bias;
			dstMat.copyTo(dst);
		}
	}

	void Classifier::train_scale(cv::InputArray data,float minVal,float maxVal){
		cv::Mat dataMat = data.getMat();
		norm_coef = cv::Mat::zeros(1, dataMat.cols, CV_32FC1);
		norm_bias = cv::Mat::zeros(1, dataMat.cols, CV_32FC1);

		float* coefLine = norm_coef.ptr<float>(0);
		float* biasLine = norm_bias.ptr<float>(0);

		for (int c = 0; c < dataMat.cols; c++){
			cv::Mat data_row = dataMat.col(c);
			double _minVal, _maxVal;
			cv::minMaxLoc(data_row, &_minVal, &_maxVal);

			float diff = _maxVal - _minVal;
			float s_diff = maxVal - minVal;

			float bias = (diff == 0.0)? 0.0 :-_minVal*s_diff / diff + minVal;
			float coef = (diff == 0.0)? 0.0 : s_diff / diff;

			coefLine[c] = coef;
			biasLine[c] = bias;
		}
	}
	std::vector<int> Classifier::getIdxVector(int sample_size,const cv::Mat& idxmat){
		assert(idxmat.empty() || idxmat.type() == CV_32SC1);

		std::vector<int> retvec;
		if (idxmat.empty()){
			retvec.reserve(sample_size);
			for (int i = 0; i < sample_size; i++){
				retvec.push_back(i);
			}
		} else{
			cv::Mat tmp = idxmat;
			tmp.copyTo(retvec);
		}
		//indexÌd¡ð­
		std::sort(retvec.begin(), retvec.end());
		retvec.erase(std::unique(retvec.begin(), retvec.end()), retvec.end());

		return retvec;
	}


}

