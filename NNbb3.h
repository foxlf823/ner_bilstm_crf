
#ifndef NNBB3_H_
#define NNBB3_H_

#include <iosfwd>
#include "Options.h"
#include "Tool.h"
#include "FoxUtil.h"
#include "Utf.h"
#include "Token.h"
#include "Sent.h"
#include <sstream>
#include "N3L.h"
#include "Document.h"
#include "EnglishPos.h"
#include "Punctuation.h"
#include "Word2Vec.h"
#include "utils.h"
#include "Example.h"

#include "Driver.h"



using namespace nr;
using namespace std;


// use relation f1 on the development set

// schema BILOU, three entity types (Bacteria,Habitat,Geographical)
#define TYPE_Bac "Bacteria"
#define TYPE_Hab "Habitat"
#define TYPE_Geo "Geographical"
#define MAX_ENTITY 13
#define B_Bacteria "B_Bacteria"
#define I_Bacteria "I_Bacteria"
#define L_Bacteria "L_Bacteria"
#define U_Bacteria "U_Bacteria"
#define B_Habitat "B_Habitat"
#define I_Habitat "I_Habitat"
#define L_Habitat "L_Habitat"
#define U_Habitat "U_Habitat"
#define B_Geographical "B_Geographical"
#define I_Geographical "I_Geographical"
#define L_Geographical "L_Geographical"
#define U_Geographical "U_Geographical"
#define OTHER "O"

int NERlabelName2labelID(const string& labelName) {
	if(labelName == B_Bacteria) {
		return 0;
	} else if(labelName == I_Bacteria) {
		return 1;
	} else if(labelName == L_Bacteria) {
		return 2;
	} else if(labelName == U_Bacteria) {
		return 3;
	} else if(labelName == B_Habitat) {
		return 4;
	} else if(labelName == I_Habitat) {
		return 5;
	} else if(labelName == L_Habitat) {
		return 6;
	} else if(labelName == U_Habitat) {
		return 7;
	} else if(labelName == B_Geographical) {
		return 8;
	} else if(labelName == I_Geographical)
		return 9;
	else if(labelName == L_Geographical)
			return 10;
	else if(labelName == U_Geographical)
			return 11;
	else
		return 12;
}

string NERlabelID2labelName(const int labelID) {
	if(labelID == 0) {
		return B_Bacteria;
	} else if(labelID == 1) {
		return I_Bacteria;
	} else if(labelID == 2) {
		return L_Bacteria;
	} else if(labelID == 3) {
		return U_Bacteria;
	} else if(labelID == 4) {
		return B_Habitat;
	} else if(labelID == 5) {
		return I_Habitat;
	} else if(labelID == 6) {
		return L_Habitat;
	} else if(labelID == 7) {
		return U_Habitat;
	} else if(labelID == 8) {
		return B_Geographical;
	} else if(labelID == 9)
		return I_Geographical;
	else if(labelID == 10)
			return L_Geographical;
	else if(labelID == 11)
			return U_Geographical;
	else
		return OTHER;
}

void outputResults(const string& id, vector<Entity>& entities, const string& dir) {
	ofstream m_outf;
	string path = dir+"/BB-event+ner-"+id+".a2";
	m_outf.open(path.c_str());

	for(int i=0;i<entities.size();i++) {

		m_outf << entities[i].id << "\t"<< entities[i].type<<" "<<entities[i].begin<<" "<<
				entities[i].end<<"\t"<<entities[i].text<<endl;

	}

	m_outf.close();
}

void loadNlpFile(const string& dirPath, vector<Document>& docs) {

	struct dirent** namelist = NULL;
	int total = scandir(dirPath.c_str(), &namelist, 0, alphasort);
	int count = 0;

	for(int i=0;i<total;i++) {

		if (namelist[i]->d_type == 8) {
			//file
			if(namelist[i]->d_name[0]=='.')
				continue;

			string filePath = dirPath;
			filePath += "/";
			filePath += namelist[i]->d_name;
			string fileName = namelist[i]->d_name;

			ifstream ifs;
			ifs.open(filePath.c_str());
			fox::Sent sent;
			string line;


			while(getline(ifs, line)) {
				if(line.empty()){
					// new line
					if(sent.tokens.size()!=0) {
						docs[count].sents.push_back(sent);
						docs[count].sents[docs[count].sents.size()-1].begin = sent.tokens[0].begin;
						docs[count].sents[docs[count].sents.size()-1].end = sent.tokens[sent.tokens.size()-1].end;
						sent.tokens.clear();
					}
				} else {
					vector<string> splitted;
					fox::split_bychar(line, splitted, '\t');
					fox::Token token;
					token.word = splitted[0];
					token.begin = atoi(splitted[1].c_str());
					token.end = atoi(splitted[2].c_str());
					token.pos = splitted[3];
					token.lemma = splitted[4];
					token.depGov = atoi(splitted[5].c_str());
					token.depType = splitted[6];
					sent.tokens.push_back(token);
				}



			}

			ifs.close();
			count++;
		}
	}

}

int parseBB3(const string& dirPath, vector<Document>& documents)
{
	struct dirent** namelist = NULL;
	int total = scandir(dirPath.c_str(), &namelist, 0, alphasort);


	for(int i=0;i<total;i++) {

		if (namelist[i]->d_type == 8) {
			//file
			if(namelist[i]->d_name[0]=='.')
				continue;

			string filePath = dirPath;
			filePath += "/";
			filePath += namelist[i]->d_name;
			string fileName = namelist[i]->d_name;

			if(string::npos != filePath.find(".a1")) { // doc
				Document doc;
				doc.id = fileName.substr(fileName.find_last_of("-")+1, fileName.find(".")-fileName.find_last_of("-")-1);
				doc.maxParagraphId = -1;
				ifstream ifs;
				ifs.open(filePath.c_str());
				string line;
				while(getline(ifs, line)) {

					if(!line.empty() && line[0]=='T') {
						if(doc.maxParagraphId < atoi(line.substr(1,1).c_str()))
							doc.maxParagraphId = atoi(line.substr(1,1).c_str());
					}
				}

				ifs.close();

				documents.push_back(doc);
			} else if(string::npos != filePath.find(".a2")) { // entity && relation
				Document& doc = documents[documents.size()-1];

				ifstream ifs;
				ifs.open(filePath.c_str());


				string line;
				while(getline(ifs, line)) {

					if(!line.empty()) {

						if(line[0] == 'T') { // entity
							vector<string> splitted;
							fox::split_bychar(line, splitted, '\t');
							Entity entity;
							entity.id = splitted[0];
							entity.text = splitted[2];

							vector<string> temp1;
							fox::split(splitted[1], temp1, " |;");

							if(temp1.size() == 3) {
								entity.type = temp1[0];
								entity.begin = atoi(temp1[1].c_str());
								entity.end = atoi(temp1[2].c_str());

								if(doc.entities.size()>0) {
									Entity& former = doc.entities[doc.entities.size()-1];
									if(isEntityOverlapped(former, entity)) {
										// if two entities is overlapped, we keep the one with narrow range
										int formerRange = former.end-former.begin;
										int entityRange = entity.end-entity.begin;
										if(entityRange<formerRange) {
											doc.entities.pop_back();
											doc.entities.push_back(entity);
										}
									} else
										doc.entities.push_back(entity);
								} else
									doc.entities.push_back(entity);

/*								cout<<"dump entity#####"<<endl;
								for(int i=0;i<doc.entities.size();i++)
									cout<<doc.entities[i].id<<endl;*/

							} else { // non-continuous entity is ignored

/*								entity.type = temp1[0];
								entity.begin = atoi(temp1[1].c_str());
								entity.end = atoi(temp1[2].c_str());
								entity.begin2 = atoi(temp1[3].c_str());
								entity.end2 = atoi(temp1[4].c_str());
								doc.entities.push_back(entity);*/
							}

						}

					}
				}
				ifs.close();

			}

		}
	}


    return 0;

}


class NNbb3 {
public:
	Options m_options;



	Driver m_driver;


  NNbb3(const Options &options):m_options(options) {

	}


	void trainAndTest(const string& trainFile, const string& devFile, const string& testFile,
			Tool& tool,
			const string& trainNlpFile, const string& devNlpFile, const string& testNlpFile) {


		// load train data
		vector<Document> trainDocuments;
		parseBB3(trainFile, trainDocuments);
		loadNlpFile(trainNlpFile, trainDocuments);

		vector<Document> devDocuments;
		if(!devFile.empty()) {
			parseBB3(devFile, devDocuments);
			loadNlpFile(devNlpFile, devDocuments);
		}
		vector<Document> testDocuments;
		if(!testFile.empty()) {
			parseBB3(testFile, testDocuments);
			loadNlpFile(testNlpFile, testDocuments);
		}

		m_driver._modelparams.labelAlpha.set_fixed_flag(false);
		m_driver._modelparams.labelAlpha.from_string(B_Bacteria);
		m_driver._modelparams.labelAlpha.from_string(I_Bacteria);
		m_driver._modelparams.labelAlpha.from_string(L_Bacteria);
		m_driver._modelparams.labelAlpha.from_string(U_Bacteria);
		m_driver._modelparams.labelAlpha.from_string(B_Habitat);
		m_driver._modelparams.labelAlpha.from_string(I_Habitat);
		m_driver._modelparams.labelAlpha.from_string(L_Habitat);
		m_driver._modelparams.labelAlpha.from_string(U_Habitat);
		m_driver._modelparams.labelAlpha.from_string(B_Geographical);
		m_driver._modelparams.labelAlpha.from_string(I_Geographical);
		m_driver._modelparams.labelAlpha.from_string(L_Geographical);
		m_driver._modelparams.labelAlpha.from_string(U_Geographical);
		m_driver._modelparams.labelAlpha.from_string(OTHER);
		m_driver._modelparams.labelAlpha.set_fixed_flag(true);
		cout << "total label size "<< m_driver._modelparams.labelAlpha.size() << endl;

		cout << "Creating Alphabet..." << endl;

		m_driver._modelparams.wordAlpha.clear();
		m_driver._modelparams.wordAlpha.from_string(unknownkey);
		m_driver._modelparams.wordAlpha.from_string(nullkey);


		createAlphabet(trainDocuments, tool);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool);
		}


		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
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
		} else {
			if(m_options.embFile.empty()) {
				assert(0);
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
		}

		initialLookupTable(m_driver._modelparams.words, &m_driver._modelparams.wordAlpha, wordEmb, m_options.wordEmbFineTune);


		m_driver._hyperparams.setRequared(m_options);
		m_driver.initial();



		vector<Example> trainExamples;
		initialTrainingExamples(tool, trainDocuments, trainExamples);
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
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	evaluateOnDev(tool, devDocuments, metric_dev, iter);

				if (metric_dev.getAccuracy() > bestDIS) {
					cout << "Exceeds best performance of " << bestDIS << endl;
					bestDIS = metric_dev.getAccuracy();

					if(testDocuments.size()>0) {

						// clear output dir
						string s = "rm -f "+m_options.output+"/*";
						system(s.c_str());

						test(tool, testDocuments);

					}
				}



		    } // devExamples > 0

		} // for iter



	}


	void initialTrainingExamples(Tool& tool, const vector<Document>& documents, vector<Example>& examples) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];

				Example eg;
				generateOneNerExample(eg, sent, doc, false);

				examples.push_back(eg);
			} // sent


		} // doc

	}

	void evaluateOnDev(Tool& tool, const vector<Document>& documents, Metric& metric_dev, int iter) {
    	metric_dev.reset();

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];
			vector<Entity> anwserEntities;


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];

				Example eg;
				generateOneNerExample(eg, sent, doc, true);

				vector<int> labelIdx;
				m_driver.predict(eg.m_features, labelIdx);


				decode(sent, labelIdx, anwserEntities);


			} // sent

			// evaluate by ourselves
			ctGoldEntity += doc.entities.size();
			ctPredictEntity += anwserEntities.size();
			for(int i=0;i<anwserEntities.size();i++) {
				int k=-1;
				int j=0;
				for(;j<doc.entities.size();j++) {
					if(anwserEntities[i].equalsBoundary(doc.entities[j])) {
						if(anwserEntities[i].equalsType(doc.entities[j])) {
							ctCorrectEntity ++;
							break;
						} else {
							k = j;
							break;
						}
					}
				}

			}


		} // doc

		metric_dev.overall_label_count = ctGoldEntity;
		metric_dev.predicated_label_count = ctPredictEntity;
		metric_dev.correct_label_count = ctCorrectEntity;
		metric_dev.print();

	}

	void test(Tool& tool, const vector<Document>& documents) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];
			vector<Entity> anwserEntities;

			int entityId = doc.maxParagraphId+1;
			int relationId = 1;

			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];

				Example eg;
				generateOneNerExample(eg, sent, doc, true);

				vector<int> labelIdx;
				m_driver.predict(eg.m_features, labelIdx);


				decode(sent, labelIdx, anwserEntities);

			} // sent

			outputResults(doc.id, anwserEntities, m_options.output);
		} // doc

	}

	void decode(const fox::Sent& sent, const vector<int>& labelIdx, vector<Entity> & anwserEntities) {

		int seq_size = labelIdx.size();
		vector<string> outputs;
		outputs.resize(seq_size);
		for (int idx = 0; idx < seq_size; idx++) {
			outputs[idx] = m_driver._modelparams.labelAlpha.from_id(labelIdx[idx], OTHER);
		}

		for(int idx=0;idx<sent.tokens.size();idx++) {
			const string& labelName = outputs[idx];
			const fox::Token& token = sent.tokens[idx];

			// decode entity label
			if(labelName == B_Bacteria || labelName == U_Bacteria ||
					labelName == B_Habitat || labelName == U_Habitat ||
					labelName == B_Geographical || labelName == U_Geographical) {
				Entity entity;
				newEntity(token, labelName, entity, 0);
				anwserEntities.push_back(entity);
			} else if(labelName == I_Bacteria || labelName == L_Bacteria ||
					labelName == I_Habitat || labelName == L_Habitat ||
					labelName == I_Geographical || labelName == L_Geographical) {
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
			if(currentLabel==I_Bacteria || currentLabel==L_Bacteria) {
				if(positionNew==-1 && labelSequence[j]==B_Bacteria) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Bacteria) {
					positionOther = j;
				}
			} else if(currentLabel==I_Habitat || currentLabel==L_Habitat) {
				if(positionNew==-1 && labelSequence[j]==B_Habitat) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Habitat) {
					positionOther = j;
				}
			} else {
				if(positionNew==-1 && labelSequence[j]==B_Geographical) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Geographical) {
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

	void generateOneNerExample(Example& eg, const fox::Sent& sent, const Document& doc, bool bPredict) {

		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
			const fox::Token& token = sent.tokens[tokenIdx];

			if(!bPredict) {
				int entityIdx = -1;
				string schemaLabel = OTHER;
				for(int i=0;i<doc.entities.size();i++) {
					const Entity& entity = doc.entities[i];

					string temp = isTokenInEntity(token, entity);
					if(temp != OTHER) {
						entityIdx = i;
						schemaLabel = temp;
						break;
					}
				}
				string labelName = entityIdx!=-1 ? schemaLabel+"_"+doc.entities[entityIdx].type : OTHER;
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


	void createAlphabet (const vector<Document>& documents, Tool& tool) {

		unordered_map<string, int> word_stat;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<documents[docIdx].sents.size();i++) {

				for(int j=0;j<documents[docIdx].sents[i].tokens.size();j++) {

					string curword = feature_word(documents[docIdx].sents[i].tokens[j]);
					word_stat[curword]++;

				}


			}


		}

		stat2Alphabet(word_stat, m_driver._modelparams.wordAlpha, "word", m_options.wordCutOff);

	}









};



#endif /* NNBB3_H_ */

