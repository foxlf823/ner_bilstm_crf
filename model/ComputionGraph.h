#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 500;

public:
	// node instances
	vector<vector<LookupNode> > word_inputs;
	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<BiNode> word_hidden2;
	vector<LinearNode> output;

	int type_num;

	//dropout nodes
	vector<vector<DropNode> > word_inputs_drop;
	vector<DropNode> word_hidden1_drop;
	vector<DropNode> word_hidden2_drop;

	// node pointers
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int typeNum){
		type_num = typeNum;
		resizeVec(word_inputs, sent_length, type_num + 1);
		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		word_hidden2.resize(sent_length);
		output.resize(sent_length);

		resizeVec(word_inputs_drop, sent_length, type_num + 1);
		word_hidden1_drop.resize(sent_length);
		word_hidden2_drop.resize(sent_length);
	}

	inline void clear(){
		Graph::clear();
		clearVec(word_inputs);
		token_repsents.clear();
		word_window.clear();
		word_hidden1.clear();
		left_lstm.clear();
		right_lstm.clear();
		word_hidden2.clear();
		output.clear();

		clearVec(word_inputs_drop);
		word_hidden1_drop.clear();
		word_hidden2_drop.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&model.words);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model.types[idy - 1]);
			}

			for (int idy = 0; idy < word_inputs[idx].size(); idy++){
				word_inputs_drop[idx][idy].setDropValue(opts.dropOut);
			}

			word_hidden1[idx].setParam(&model.tanh1_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden1_drop[idx].setDropValue(opts.dropOut);

			word_hidden2[idx].setParam(&model.tanh2_project);
			word_hidden2[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden2_drop[idx].setDropValue(opts.dropOut);
		}
		word_window.setContext(opts.wordcontext);
		left_lstm.setParam(&model.left_lstm_project, opts.dropOut, true);
		right_lstm.setParam(&model.right_lstm_project, opts.dropOut, false);

		for (int idx = 0; idx < output.size(); idx++){
			output[idx].setParam(&model.olayer_linear);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx][0].forward(this, feature.words[0]);

			//drop out
			word_inputs_drop[idx][0].forward(this, &word_inputs[idx][0]);

			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(this, feature.types[idy - 1]);
				//drop out
				word_inputs_drop[idx][idy].forward(this, &word_inputs[idx][idy]);
			}

			token_repsents[idx].forward(this, getPNodes(word_inputs_drop[idx], word_inputs_drop[idx].size()));
		}

		//windowlized
		word_window.forward(this, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(this, &(word_window._outputs[idx]));

			word_hidden1_drop[idx].forward(this, &word_hidden1[idx]);
		}

		left_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size));
		right_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden2[idx].forward(this, &(left_lstm._hiddens_drop[idx]), &(right_lstm._hiddens_drop[idx]));

			word_hidden2_drop[idx].forward(this, &word_hidden2[idx]);

			output[idx].forward(this, &(word_hidden2_drop[idx]));
		}
	}

};

#endif /* SRC_ComputionGraph_H_ */
