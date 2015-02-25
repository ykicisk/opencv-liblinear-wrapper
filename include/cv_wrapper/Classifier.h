#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <iostream>
#include <iomanip>
#include <functional>

#define ABSTRUCT_CLONEABLE(T) \
	protected:\
	virtual T* _cloneinner()const=0;\
	public:\
	std::shared_ptr<T> clone()const{ \
		return std::shared_ptr<T>(_cloneinner()); \
	}

#define CLONEABLE(T) \
	protected:\
	virtual T* _cloneinner()const{ \
		return new T(*this); \
	}\
	public:\
	std::shared_ptr<T>clone()const{ \
		return std::shared_ptr<T>(_cloneinner()); \
	}

namespace cv_wrapper{
	struct ClassifierParam {
		std::string name;
	};

	class Classifier;
	typedef std::shared_ptr<Classifier> SClassifier;

	class Classifier {
	protected:
		bool trained_flag;
		cv::Mat norm_coef;
		cv::Mat norm_bias;

		Classifier(){}
		std::string name;
		Classifier(const std::string& name)
			:name(name)
		{ };

		Classifier(const Classifier& obj){
			trained_flag = false;
			if (!obj.norm_coef.empty()) norm_coef = obj.norm_coef.clone();
			if (!obj.norm_bias.empty()) norm_bias = obj.norm_bias.clone();
		}

	public:
		bool isTrained(){return trained_flag;}
		virtual ~Classifier(){};
		static SClassifier create(){ 
			throw std::runtime_error("no create function!"); 
		}

		void reset_scale();
		void train_scale(cv::InputArray data,float minVal=-1.0,float maxVal=1.0);
		void normalize(cv::InputArray src, cv::OutputArray dst);

		virtual void train(cv::InputArray labeled,cv::InputArray label,cv::InputArray sampleIdx=cv::Mat(),cv::InputArray weight=cv::Mat())=0;

		virtual float predict(const cv::Mat& data)=0;
		virtual std::map<float,float> predict_probability(const cv::Mat& data)=0;

		static std::vector<int> getIdxVector(int sample_size,const cv::Mat& idxmat);

		std::string getName()const{
			return name;
		}

		ABSTRUCT_CLONEABLE(Classifier);
	};

}
