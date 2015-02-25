#include "cv_wrapper/LibLinearClassifier.h"
#include <cassert>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include <boost/filesystem.hpp>

namespace cv_wrapper {

	LibLinearClassifier::~LibLinearClassifier(){
		if (trained_flag){
			if(model != NULL) free_and_destroy_model(&model);
		}
	}

	void LibLinearClassifier::train(cv::InputArray data, cv::InputArray label, cv::InputArray sampleIdx,cv::InputArray weight)
	{
		cv::Mat dataMat = data.getMat();
		std::vector<float> labelvec = label.getMat();
		std::vector<int> idxvec = getIdxVector(dataMat.rows, sampleIdx.getMat());
		
		//scaling
		normalize(dataMat, dataMat);

		std::vector<float> class_label;
		class_label.reserve(idxvec.size());

		std::vector<float> weightvec = weight.getMat();
		for (auto it = idxvec.begin(); it != idxvec.end();){
			if (!weightvec.empty() && weightvec[*it] == 0.0){
				it = idxvec.erase(it);
			} else{
				class_label.push_back(labelvec[*it]);
				++it;
			}
		}
		std::sort(class_label.begin(),class_label.end());
		class_label.erase( unique(class_label.begin(), class_label.end()), class_label.end() );

		// load model file
		if (!param.save_path.empty()){
			namespace fs = boost::filesystem;
			if (fs::exists(param.save_path)){
				model = LibLinear::load_model(param.save_path.c_str());
				trained_flag = true;
				return;
			}
		}

		assert(weightvec.empty() || dataMat.rows == weightvec.size());
		assert(dataMat.type()==CV_32FC1);

		//sample weight
		if (weightvec.empty()){
			weightvec.reserve(dataMat.rows);
			for (int i = 0; i < dataMat.rows; i++){
				weightvec.push_back(1.0);
			}
		}

		//class weight
		{
			param.p.nr_weight = class_label.size();
			param.p.weight_label = new int[param.p.nr_weight];
			for(int i=0;i < param.p.nr_weight;i++){
				param.p.weight_label[i] = class_label[i];
			}

			param.p.weight = new double[param.p.nr_weight];
			if(param.class_weight.empty()){
				for(int i=0;i < param.p.nr_weight;i++) param.p.weight[i] = 1.0;
			}else{
				for(int i=0;i < param.p.nr_weight;i++){
					int l = class_label[i];
					if(param.class_weight.count(l) == 0) throw std::runtime_error("no class weight");
					param.p.weight[i] = param.class_weight[l];
				}
			}
		}

		//learning!!
		{
			LibLinear::problem prob;
			LibLinear::feature_node *prob_vec;

			if (trained_flag){
				free_and_destroy_model(&model);
			}

			const int MAX_DIM = dataMat.cols;
			const int MAX_SAMPLE = idxvec.size();

			prob.bias = param.bias; 
			prob.n = MAX_DIM+1;
			prob.l = MAX_SAMPLE;
			prob.y = new double[prob.l]; //label
			prob.x = new LibLinear::feature_node *[prob.l]; //->prob_vec
			prob.W = new double[prob.l]; //weight
			prob_vec = new LibLinear::feature_node[(MAX_DIM + 2)*prob.l]; //feature

			param.prob_n = prob.n;
			param.prob_l = prob.n;

			for (int i = 0; i < prob.l; i++){
				int idx = idxvec[i];
				prob.x[i] = &prob_vec[(MAX_DIM + 2)*i];//pointer
				prob.y[i] = labelvec[idx];//label
				prob.W[i] = weightvec[idx];//weight

				float * fline = dataMat.ptr<float>(idx);
				for (int j = 0; j < MAX_DIM; j++){
					prob.x[i][j].index = j + 1;
					prob.x[i][j].value = fline[j];
				}

				//bias : no bias = 0.0
				prob.x[i][MAX_DIM].index = MAX_DIM + 1;
				prob.x[i][MAX_DIM].value = (prob.bias >= 0) ? prob.bias : 0.0;

				//end
				prob.x[i][MAX_DIM + 1].index = -1;
			}

			model = LibLinear::train(&prob, &param.p);
			trained_flag = true;

			delete[] prob.y;
			delete[] prob.x;
			delete[] prob.W;
			delete[] prob_vec;
		}

		delete[]param.p.weight;
		delete[]param.p.weight_label;

		if (!param.save_path.empty()) LibLinear::save_model(param.save_path.c_str(), model);
	}

	float LibLinearClassifier::predict(const cv::Mat& data)
	{
		float label=-1.0;
		if (data.empty()) return label;

		assert(data.type() == CV_32FC1);

		//scaling
		cv::Mat _data;
		normalize(data, _data);

		//classification
		{
			LibLinear::feature_node *node = new LibLinear::feature_node[(param.prob_n + 1)];
			const float *dataline = _data.ptr<float>(0);
			for (int j = 0; j < param.prob_n - 1; j++){
				node[j].index = j + 1;
				node[j].value = dataline[j];
			}

			node[param.prob_n - 1].index = param.prob_n;//bias
			node[param.prob_n - 1].value = (param.bias >= 0) ? param.bias : 0.0;//bias

			node[param.prob_n].index = -1;//end

			label = LibLinear::predict(model, node);
			delete[] node;
		}
		return label;
	}
	std::map<float,float> LibLinearClassifier::predict_probability(const cv::Mat& data)
	{
		std::map<float,float> ret;
		if (data.empty()) return ret;

		assert(data.type() == CV_32FC1);

		//scaling
		cv::Mat _data;
		normalize(data, _data);

		//classification
		{
			LibLinear::feature_node *node = new LibLinear::feature_node[(param.prob_n + 1)];
			const float *dataline = _data.ptr<float>(0);
			for (int j = 0; j < param.prob_n - 1; j++){
				node[j].index = j + 1;
				node[j].value = dataline[j];
			}

			node[param.prob_n - 1].index = param.prob_n;//bias
			node[param.prob_n - 1].value = (param.bias >= 0) ? param.bias : 0.0;//bias

			node[param.prob_n].index = -1;//end

			int nr_class = LibLinear::get_nr_class(model);
			double* estimates = new double[nr_class];
			LibLinear::predict_probability(model, node, estimates);

			for(int i=0;i<nr_class;i++)
			{
				int l = model->label[i];
				ret[l] = estimates[i];
			}

			delete[] node;
			delete[] estimates;
		}

		return ret;
	}
}
