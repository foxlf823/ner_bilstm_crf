
#ifndef NNgm_H_
#define NNgm_H_

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


// use relation f1 on the development set

// schema BILOU, one entity types (Gene)
#define MAX_ENTITY 5
#define B "B"
#define I "I"
#define L "L"
#define U "U"
#define O "O"

int NERlabelName2labelID(const string& labelName) {
	if(labelName == B) {
		return 0;
	} else if(labelName == I) {
		return 1;
	} else if(labelName == L) {
		return 2;
	} else if(labelName == U) {
		return 3;
	}
	else
		return 4;
}

string NERlabelID2labelName(const int labelID) {
	if(labelID == 0) {
		return B;
	} else if(labelID == 1) {
		return I;
	} else if(labelID == 2) {
		return L;
	} else if(labelID == 3) {
		return U;
	}
	else
		return O;
}

int parse(const string& sentFile, const string& entityFile, Tool& tool, vector<GMsentence>& sentences)
{

	ifstream sent_ifs;
	sent_ifs.open(sentFile.c_str());
	ifstream entity_ifs;
	entity_ifs.open(entityFile.c_str());

	string line;

	while(getline(sent_ifs, line)) {
		if(!line.empty()) {
			GMsentence sentence;
			int pos_firstwhitespace = line.find_first_of(' ');
			sentence.id = line.substr(0, pos_firstwhitespace);

			sentence.text = line.substr(pos_firstwhitespace+1);

			vector<fox::Token> tokens;
			tool.tokenizer.tokenize(0, sentence.text, sentence.tokens);

			sentences.push_back(sentence);
		}

	}

	while(getline(entity_ifs, line)) {
		if(!line.empty()) {
			vector<string> splitted;
			fox::split_bychar(line, splitted, '|');

			string id = splitted[0];

			vector<string> boundary;
			fox::split_bychar(splitted[1], boundary, ' ');
			int start = atoi(boundary[0].c_str());
			int end = atoi(boundary[1].c_str());

			int i=0;
			for(;i<sentences.size();i++) {
				GMsentence & sentence = sentences[i];
				if(id == sentence.id)
					break;
			}
			assert(i<sentences.size());

			int adjustedStart = -1;
			int adjustedEnd = -1;
			GMsentence & sentence = sentences[i];
			int countWhiteSpace=0;
			for(int j=0; j<sentence.text.length(); j++) {
				char ch = sentence.text.at(j);

				if(j-countWhiteSpace == start && ch!=' ') {
					adjustedStart = j;
				}
				if(j-countWhiteSpace == end && ch!=' ') {
					adjustedEnd = j;
				}


				if(ch == ' ') {
					countWhiteSpace++;
				}

				if(adjustedStart!=-1 && adjustedEnd!=-1)
					break;
			}

			assert(adjustedStart!=-1);
			assert(adjustedEnd!=-1);

			Entity entity;
			entity.begin = adjustedStart;
			entity.end = adjustedEnd+1;
			entity.text = splitted[2];

			sentence.entities.push_back(entity);

		}
	}

	sent_ifs.close();
	entity_ifs.close();

    return 0;

}

void outputResults(const GMsentence& sentence, const vector<Entity>& entities, const string& file) {
	ofstream m_outf;
	m_outf.open(file.c_str(), ios::app);

	for(int i=0;i<entities.size();i++) {
		const Entity& entity = entities[i];

		int adjustedStart = -1;
		int adjustedEnd = -1;
		int countWhiteSpace=0;

		for(int j=0; j<sentence.text.length(); j++) {
			char ch = sentence.text.at(j);

			if(j == entity.begin) {
				adjustedStart = entity.begin-countWhiteSpace;
			}
			if(j == entity.end-1) {
				adjustedEnd = entity.end-1-countWhiteSpace;
			}

			if(ch == ' ') {
				countWhiteSpace++;
			}

			if(adjustedStart!=-1 && adjustedEnd!=-1)
				break;
		}


		m_outf << sentence.id <<"|"<< adjustedStart << " "<< adjustedEnd<<"|"<<entity.text<<endl;

	}

	m_outf.close();
}


class NNgm {
public:
	Options m_options;



	Driver m_driver;


  NNgm(const Options &options):m_options(options) {

	}


	void trainAndTest(const string& trainFile, const string& testFile,
			Tool& tool,
			const string& trainAnswerFile, const string& testAnswerFile, const vector<string> & kbFiles) {


		// load train data
		vector<GMsentence> trainSentences;
		parse(trainFile, trainAnswerFile, tool, trainSentences);


		vector<GMsentence> testSentences;
		if(!testFile.empty()) {
			parse(testFile, testAnswerFile, tool, testSentences);
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

