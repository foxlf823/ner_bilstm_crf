
#ifndef NNgm3_H_
#define NNgm3_H_

#include <iosfwd>
#include "Options.h"
#include "Tool.h"
#include "FoxUtil.h"
#include "Utf.h"
#include "Token.h"
#include <sstream>
#include "N3L.h"
#include "Word2Vec.h"
#include "utils.h"
#include "Example.h"
#include "GMsentence.h"

#include "Driver.h"



using namespace nr;
using namespace std;


// compare with banner 7500->7500
// use relation f1 on the development set




class NNgm3 {
public:
	Options m_options;



	Driver m_driver;


  NNgm3(const Options &options):m_options(options) {

	}


	void trainAndTest(const string& trainFile, const string& testFile,
			Tool& tool,
			const string& trainAnswerFile, const string& testAnswerFile, const vector<string> & kbFiles) {


		// load train data
		vector<GMsentence> totalSentences;
		parse(trainFile, trainAnswerFile, tool, totalSentences);

		vector<GMsentence> trainSentences;
		for(int i=0;i<7500;i++) {
			trainSentences.push_back(totalSentences[i]);
		}
		vector<GMsentence> testSentences;
		for(int i=7500;i<15000;i++) {
			testSentences.push_back(totalSentences[i]);
		}

		m_driver._modelparams.labelAlpha.set_fixed_flag(false);
		m_driver._modelparams.labelAlpha.from_string(B);
		m_driver._modelparams.labelAlpha.from_string(I);
		m_driver._modelparams.labelAlpha.from_string(L);
		m_driver._modelparams.labelAlpha.from_string(U);
		m_driver._modelparams.labelAlpha.from_string(O);
		m_driver._modelparams.labelAlpha.set_fixed_flag(true);
		cout << "total label size "<< m_driver._modelparams.labelAlpha.size() << endl;

		cout << "Creating Alphabet..." << endl;

		m_driver._modelparams.wordAlpha.clear();
		m_driver._modelparams.wordAlpha.from_string(unknownkey);
		m_driver._modelparams.wordAlpha.from_string(nullkey);


		createAlphabet(trainSentences, tool);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!testSentences.empty())
				createAlphabet(testSentences, tool);
		}


		NRMat<dtype> wordEmb;

		if(m_options.embFile.empty()) {
			cout<<"random emb"<<endl;

			randomInitNrmat(wordEmb, m_driver._modelparams.wordAlpha, m_options.wordEmbSize, 1000, m_options.initRange);
		} else {
			cout<< "load pre-trained emb"<<endl;
			tool.w2v->loadFromBinFile(m_options.embFile, false, true);

			double* emb = new double[m_driver._modelparams.wordAlpha.size()*m_options.wordEmbSize];
			fox::initArray2((double *)emb, (int)m_driver._modelparams.wordAlpha.size(), m_options.wordEmbSize, 0.0);
			vector<string> known;
			map<string, int> IDs;
			alphabet2vectormap(m_driver._modelparams.wordAlpha, known, IDs);

			tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

			wordEmb.resize(m_driver._modelparams.wordAlpha.size(), m_options.wordEmbSize);
			array2NRMat((double*) emb, m_driver._modelparams.wordAlpha.size(), m_options.wordEmbSize, wordEmb);

			delete[] emb;
		}


		initialLookupTable(m_driver._modelparams.words, &m_driver._modelparams.wordAlpha, wordEmb, m_options.wordEmbFineTune);


		m_driver._hyperparams.setRequared(m_options);
		m_driver.initial();



		vector<Example> trainExamples;
		initialTrainingExamples(tool, trainSentences, trainExamples);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;


		dtype bestDIS = 0;
		int inputSize = trainExamples.size();
		int batchBlock = inputSize / m_options.batchSize;
		if (inputSize % m_options.batchSize != 0)
			batchBlock++;

		std::vector<int> indexes;
		for (int i = 0; i < inputSize; ++i)
			indexes.push_back(i);

		static Metric eval, metric_dev;
		static vector<Example> subExamples;


		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

		    eval.reset();


		    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				int start_pos = updateIter * m_options.batchSize;
				int end_pos = (updateIter + 1) * m_options.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;

				for (int idy = start_pos; idy < end_pos; idy++) {
					subExamples.push_back(trainExamples[indexes[idy]]);
				}

				int curUpdateIter = iter * batchBlock + updateIter;
				dtype cost = m_driver.train(subExamples, curUpdateIter);
				//m_driver.checkgrad(subExamples, curUpdateIter + 1);

				eval.overall_label_count += m_driver._eval.overall_label_count;
				eval.correct_label_count += m_driver._eval.correct_label_count;

/*				if ((curUpdateIter + 1) % m_options.verboseIter == 0) {

					std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
					std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
				}*/
				m_driver.updateModel();
		    }

		    // an iteration end, begin to evaluate
		    if (testSentences.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	evaluateOnDev(tool, testSentences, metric_dev);

				if (metric_dev.getAccuracy() > bestDIS) {
					cout << "Exceeds best performance of " << bestDIS << endl;
					bestDIS = metric_dev.getAccuracy();

					if(testSentences.size()>0) {


						test(tool, testSentences, metric_dev);

					}
				}



		    } // devExamples > 0

		} // for iter



	}


	void initialTrainingExamples(Tool& tool, const vector<GMsentence>& sentences, vector<Example>& examples) {

		for(int sentIdx=0;sentIdx<sentences.size();sentIdx++) {
			const GMsentence& sentence = sentences[sentIdx];

			Example eg;
			generateOneNerExample(eg, sentence, false);

			examples.push_back(eg);

		} // doc

	}

	void evaluateOnDev(Tool& tool, const vector<GMsentence>& sentences, Metric& metric_dev) {
    	metric_dev.reset();

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;


		for(int sentIdx=0;sentIdx<sentences.size();sentIdx++) {
			const GMsentence & sentence = sentences[sentIdx];

			vector<Entity> anwserEntities;

			Example eg;
			generateOneNerExample(eg, sentence, true);

			vector<int> labelIdx;
			m_driver.predict(eg.m_features, labelIdx);


			decode(sentence, labelIdx, anwserEntities);


			// evaluate by ourselves
			ctGoldEntity += sentence.entities.size();
			ctPredictEntity += anwserEntities.size();
			for(int i=0;i<anwserEntities.size();i++) {

				int j=0;
				for(;j<sentence.entities.size();j++) {
					if(anwserEntities[i].equalsBoundary(sentence.entities[j])) {
						ctCorrectEntity ++;
						break;

					}
				}

			}


		} // sent


		metric_dev.overall_label_count = ctGoldEntity;
		metric_dev.predicated_label_count = ctPredictEntity;
		metric_dev.correct_label_count = ctCorrectEntity;
		metric_dev.print();

	}

	void test(Tool& tool, const vector<GMsentence>& sentences, Metric& metric_dev) {
		metric_dev.reset();

		string s = "rm -f "+m_options.output;
		system(s.c_str());

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;


		for(int sentIdx=0;sentIdx<sentences.size();sentIdx++) {
			const GMsentence & sentence = sentences[sentIdx];

			vector<Entity> anwserEntities;

			Example eg;
			generateOneNerExample(eg, sentence, true);

			vector<int> labelIdx;
			m_driver.predict(eg.m_features, labelIdx);


			decode(sentence, labelIdx, anwserEntities);


			// evaluate by ourselves
			ctGoldEntity += sentence.entities.size();
			ctPredictEntity += anwserEntities.size();
			for(int i=0;i<anwserEntities.size();i++) {
				int j=0;
				for(;j<sentence.entities.size();j++) {
					if(anwserEntities[i].equalsBoundary(sentence.entities[j])) {

						ctCorrectEntity ++;
						break;

					}
				}

			}

			outputResults(sentence, anwserEntities, m_options.output);

		} // sent


		metric_dev.overall_label_count = ctGoldEntity;
		metric_dev.predicated_label_count = ctPredictEntity;
		metric_dev.correct_label_count = ctCorrectEntity;
		metric_dev.print();



	}

	void decode(const GMsentence& sent, const vector<int>& labelIdx, vector<Entity> & anwserEntities) {

		int seq_size = labelIdx.size();
		vector<string> outputs;
		outputs.resize(seq_size);
		for (int idx = 0; idx < seq_size; idx++) {
			outputs[idx] = m_driver._modelparams.labelAlpha.from_id(labelIdx[idx], O);
		}

		for(int idx=0;idx<sent.tokens.size();idx++) {
			const string& labelName = outputs[idx];
			const fox::Token& token = sent.tokens[idx];

			// decode entity label
			if(labelName == B || labelName == U) {
				Entity entity;
				newEntity(token, labelName, entity, 0);
				anwserEntities.push_back(entity);
			} else if(labelName == I || labelName == L) {
				if(checkWrongState(outputs, idx+1)) {
					Entity& entity = anwserEntities[anwserEntities.size()-1];
					appendEntity(token, entity);
				}
			}

		} // token
	}

	// Only used when current label is I or L, check state from back to front
	// in case that "O I I", etc.
	bool checkWrongState(const vector<string>& labelSequence, int size) {
		int positionNew = -1; // the latest type-consistent B
		int positionOther = -1; // other label except type-consistent I

		const string& currentLabel = labelSequence[size-1];

		for(int j=size-2;j>=0;j--) {
			if(currentLabel==I || currentLabel==L) {
				if(positionNew==-1 && labelSequence[j]==B) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I) {
					positionOther = j;
				}
			}

			if(positionOther!=-1 && positionNew!=-1)
				break;
		}

		if(positionNew == -1)
			return false;
		else if(positionOther<positionNew)
			return true;
		else
			return false;
	}

	void generateOneNerExample(Example& eg, const GMsentence& sent, bool bPredict) {

		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
			const fox::Token& token = sent.tokens[tokenIdx];

			if(!bPredict) {
				int entityIdx = -1;
				string schemaLabel = O;
				for(int i=0;i<sent.entities.size();i++) {
					const Entity& entity = sent.entities[i];

					string temp = isTokenInEntity(token, entity);
					if(temp != O) {
						entityIdx = i;
						schemaLabel = temp;
						break;
					}
				}
				string labelName = entityIdx!=-1 ? schemaLabel : O;
				int labelID = NERlabelName2labelID(labelName);
				vector<dtype> labelsForThisToken;
				for(int i=0;i<MAX_ENTITY;i++)
					labelsForThisToken.push_back(0.0);
				labelsForThisToken[labelID] = 1.0;

				eg.m_labels.push_back(labelsForThisToken);
			}

			Feature featureForThisToken;
			featureForThisToken.words.push_back(feature_word(token));

			eg.m_features.push_back(featureForThisToken);

		} // token


	}


	void createAlphabet (const vector<GMsentence>& sentences, Tool& tool) {

		unordered_map<string, int> word_stat;

		for(int i=0;i<sentences.size();i++) {

			for(int j=0;j<sentences[i].tokens.size();j++) {

				string curword = feature_word(sentences[i].tokens[j]);
				word_stat[curword]++;

			}


		}

		stat2Alphabet(word_stat, m_driver._modelparams.wordAlpha, "word", m_options.wordCutOff);

	}









};



#endif /* NNBB3_H_ */

