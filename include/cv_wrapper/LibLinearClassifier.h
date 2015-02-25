#pragma once
#include "cv_wrapper/Classifier.h"
#include "linear.h"

namespace cv_wrapper{

	class LibLinearClassifier:public Classifier {
	public:
		struct Param : public ClassifierParam{
			int prob_n;
			int prob_l;
			double bias = -1;

			std::map<int,double> class_weight;
			LibLinear::parameter p;
			std::string save_path="";

			Param(){
				name = "LibLinear";
			}
		}param;
	protected:
		LibLinear::model *model;

		LibLinearClassifier(const Param& p)
			:Classifier(p.name),param(p)
		{};
	public:
		virtual ~LibLinearClassifier();
		LibLinearClassifier(const LibLinearClassifier& obj)
			:Classifier(obj)
		{
			param = obj.param;
			trained_flag = false;
		}

		static SClassifier create(const Param &params = Param()){
			return SClassifier(new LibLinearClassifier(params));
		}
		void train(cv::InputArray labeled,cv::InputArray label,cv::InputArray sampleIdx=cv::Mat(),cv::InputArray weight=cv::Mat());

		double predict(const cv::Mat& data)override;
		std::map<double,double> predict_probability(const cv::Mat& data)override;

		CLONEABLE(LibLinearClassifier);
	};
}
