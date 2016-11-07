#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside

	vector<Alphabet> typeAlphas; // should be initialized outside
	vector<LookupTable> types;  // should be initialized outside


	LSTM1Params left_lstm_project; //left lstm
	LSTM1Params right_lstm_project; //right lstm
	UniParams tanh1_project; // hidden
	BiParams tanh2_project; // hidden
	UniParams olayer_linear; // output


public:
	Alphabet labelAlpha; // should be initialized outside
	CRFMLLoss loss;


public:
	bool initial(HyperParams& opts){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordwindow = 2 * opts.wordcontext + 1;
		opts.wordDim = words.nDim;
		opts.unitsize = opts.wordDim;
		opts.typeDims.clear();
		for (int idx = 0; idx <types.size(); idx++){
/*			if (types[idx].nVSize <= 0 || typeAlphas[idx].size() <= 0){
				return false;
			}*/
			opts.typeDims.push_back(types[idx].nDim);
			opts.unitsize += opts.typeDims[idx];
		}
		opts.labelSize = labelAlpha.size();
		opts.inputsize = opts.wordwindow * opts.unitsize;

		tanh1_project.initial(opts.hiddensize, opts.inputsize, true);
		left_lstm_project.initial(opts.rnnhiddensize, opts.hiddensize);
		right_lstm_project.initial(opts.rnnhiddensize, opts.hiddensize);
		tanh2_project.initial(opts.hiddensize, opts.rnnhiddensize, opts.rnnhiddensize, true);
		olayer_linear.initial(opts.labelSize, opts.hiddensize, false);

		loss.initial(opts.labelSize);

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		for (int idx = 0; idx < types.size(); idx++){
			types[idx].exportAdaParams(ada);
		}
		tanh1_project.exportAdaParams(ada);
		left_lstm_project.exportAdaParams(ada);
		right_lstm_project.exportAdaParams(ada);
		tanh2_project.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
		loss.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(words.E), "_words.E");
		for (int idx = 0; idx < types.size(); idx++){
			stringstream ss;
			ss << "types[" << idx << "].E";
			checkgrad.add(&(types[idx].E), ss.str());
		}
		checkgrad.add(&(tanh1_project.W), "tanh1_project.W");
		checkgrad.add(&(tanh1_project.b), "tanh1_project.b");

		checkgrad.add(&(left_lstm_project.input.W1), "left_lstm_project.input.W1");
		checkgrad.add(&(left_lstm_project.input.W2), "left_lstm_project.input.W2");
		checkgrad.add(&(left_lstm_project.input.b), "left_lstm_project.input.b");
		checkgrad.add(&(left_lstm_project.output.W1), "left_lstm_project.output.W1");
		checkgrad.add(&(left_lstm_project.output.W2), "left_lstm_project.output.W2");
		checkgrad.add(&(left_lstm_project.output.b), "left_lstm_project.output.b");
		checkgrad.add(&(left_lstm_project.forget.W1), "left_lstm_project.forget.W1");
		checkgrad.add(&(left_lstm_project.forget.W2), "left_lstm_project.forget.W2");
		checkgrad.add(&(left_lstm_project.forget.b), "left_lstm_project.forget.b");
		checkgrad.add(&(left_lstm_project.cell.W1), "left_lstm_project.cell.W1");
		checkgrad.add(&(left_lstm_project.cell.W2), "left_lstm_project.cell.W2");
		checkgrad.add(&(left_lstm_project.cell.b), "left_lstm_project.cell.b");

		checkgrad.add(&(right_lstm_project.input.W1), "right_lstm_project.input.W1");
		checkgrad.add(&(right_lstm_project.input.W2), "right_lstm_project.input.W2");
		checkgrad.add(&(right_lstm_project.input.b), "right_lstm_project.input.b");
		checkgrad.add(&(right_lstm_project.output.W1), "right_lstm_project.output.W1");
		checkgrad.add(&(right_lstm_project.output.W2), "right_lstm_project.output.W2");
		checkgrad.add(&(right_lstm_project.output.b), "right_lstm_project.output.b");
		checkgrad.add(&(right_lstm_project.forget.W1), "right_lstm_project.forget.W1");
		checkgrad.add(&(right_lstm_project.forget.W2), "right_lstm_project.forget.W2");
		checkgrad.add(&(right_lstm_project.forget.b), "right_lstm_project.forget.b");
		checkgrad.add(&(right_lstm_project.cell.W1), "right_lstm_project.cell.W1");
		checkgrad.add(&(right_lstm_project.cell.W2), "right_lstm_project.cell.W2");
		checkgrad.add(&(right_lstm_project.cell.b), "right_lstm_project.cell.b");


		checkgrad.add(&(tanh2_project.W1), "tanh2_project.W1");
		checkgrad.add(&(tanh2_project.W2), "tanh2_project.W2");
		checkgrad.add(&(tanh2_project.b), "tanh2_project.b");

		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
		checkgrad.add(&(olayer_linear.b), "olayer_linear.b");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */
