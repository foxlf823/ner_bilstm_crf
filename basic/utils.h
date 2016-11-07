
#ifndef UTILS_H_
#define UTILS_H_


#include <stdio.h>
#include <vector>
#include "Word2Vec.h"
#include "Utf.h"
#include "Entity.h"
#include "Token.h"
#include "FoxUtil.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "Document.h"
#include <list>
#include <sstream>
#include "N3L.h"
#include "Term.h"
#include "Tool.h"

using namespace nr;
using namespace std;


void loadKb(const string& filePath, Tool & tool, vector<Term> & kb) {
	ifstream ifs;
	ifs.open(filePath.c_str());
	string line;

	while(getline(ifs, line)) {
		if(!line.empty()){
			Term t;
			tool.tokenizer.tokenize(0, line, t.name);

			kb.push_back(t);
		}


	}

	ifs.close();
}



void appendEntity(const fox::Token& token, Entity& entity) {
	int whitespacetoAdd = token.begin-entity.end;
	for(int j=0;j<whitespacetoAdd;j++)
		entity.text += " ";
	entity.text += token.word;
	entity.end = token.end;
}

void newEntity(const fox::Token& token, const string& labelName, Entity& entity, int entityId) {
	stringstream ss;
	ss<<"T"<<entityId;
	entity.id = ss.str();
	entity.type = labelName.substr(labelName.find("_")+1);
	entity.begin = token.begin;
	entity.end = token.end;
	entity.text = token.word;
}







int findEntityById(const string& id, const vector<Entity>& entities) {

	for(int i=0;i<entities.size();i++) {
		if(entities[i].id==id) {
			return i;
		}

	}

	return -1;
}



bool isEntityOverlapped(const Entity& former, const Entity& latter) {
	if(former.end2==-1) {
		if(former.end<=latter.begin)
			return false;
		else
			return true;
	} else {
		if(former.end2<=latter.begin)
			return false;
		else
			return true;
	}
}





bool isTokenBeforeEntity(const fox::Token& tok, const Entity& entity) {
	if(tok.begin<entity.begin)
		return true;
	else
		return false;
}

bool isTokenAfterEntity(const fox::Token& tok, const Entity& entity) {
	if(entity.end2 == -1) {
		if(tok.end>entity.end)
			return true;
		else
			return false;

	} else {
		if(tok.end>entity.end2)
			return true;
		else
			return false;

	}

}


string isTokenInEntity(const fox::Token& tok, const Entity& entity) {

	if(tok.begin==entity.begin && tok.end==entity.end)
		return "U";
	else if(tok.begin==entity.begin)
		return "B";
	else if(tok.end==entity.end)
		return "L";
	else if(tok.begin>entity.begin && tok.end<entity.end)
		return "I";
	else
		return "O";

}

bool boolTokenInEntity(const fox::Token& tok, const Entity& entity) {

	if(entity.end2 == -1) {
		if(tok.begin>=entity.begin && tok.end<=entity.end)
			return true;
		else
			return false;
	} else {

		if((tok.begin>=entity.begin && tok.end<=entity.end) ||
				(tok.begin>=entity.begin2 && tok.end<=entity.end2))
			return true;
		else
			return false;

	}
}

bool isTokenBetweenTwoEntities(const fox::Token& tok, const Entity& former, const Entity& latter) {

	if(former.end2 == -1) {
		if(tok.begin>=former.end && tok.end<=latter.begin)
			return true;
		else
			return false;
	} else {
		if(tok.begin>=former.end2 && tok.end<=latter.begin)
			return true;
		else
			return false;
	}
}


// sentence spans from begin(include) to end(exclude), sorted because doc.entities are sorted
void findEntityInSent(int begin, int end, const Document& doc, vector<Entity>& results) {

	for(int i=0;i<doc.entities.size();i++) {
		if(doc.entities[i].begin2==-1) {
			if(doc.entities[i].begin >= begin && doc.entities[i].end <= end)
				results.push_back(doc.entities[i]);
		} else {
			if(doc.entities[i].begin >= begin && doc.entities[i].end2 <= end)
				results.push_back(doc.entities[i]);
		}

	}

	return;
}

void findEntityInSent(int begin, int end, const vector<Entity>& source, vector<Entity>& results) {

	for(int i=0;i<source.size();i++) {
		if(source[i].begin2==-1) {
			if(source[i].begin >= begin && source[i].end <= end)
				results.push_back(source[i]);
		} else {
			if(source[i].begin >= begin && source[i].end2 <= end)
				results.push_back(source[i]);
		}

	}

	return;
}

void deleteEntity(vector<Entity>& entities, const Entity& target)
{
	vector<Entity>::iterator iter = entities.begin();
	for(;iter!=entities.end();iter++) {
		if((*iter).equals(target)) {
			break;
		}
	}
	if(iter!=entities.end()) {
		entities.erase(iter);
	}
}

int containsEntity(vector<Entity>& source, const Entity& target) {

	for(int i=0;i<source.size();i++) {
		if(source[i].equals(target))
			return i;

	}

	return -1;
}



double precision(int correct, int predict) {
	return 1.0*correct/predict;
}

double recall(int correct, int gold) {
	return 1.0*correct/gold;
}

double f1(int correct, int gold, int predict) {
	double p = precision(correct, predict);
	double r = recall(correct, gold);

	return 2*p*r/(p+r);
}

void alphabet2vectormap(const Alphabet& alphabet, vector<string>& vector, map<string, int>& IDs) {

	for (int j = 0; j < alphabet.size(); ++j) {
		string str = alphabet.from_id(j);
		vector.push_back(str);
		IDs.insert(map<string, int>::value_type(str, j));
	}

}

template<typename T>
void array2NRMat(T * array, int sizeX, int sizeY, NRMat<T>& mat) {
	for(int i=0;i<sizeX;i++) {
		for(int j=0;j<sizeY;j++) {
			mat[i][j] = *(array+i*sizeY+j);
		}
	}
}


void initialLookupTable(LookupTable& table, PAlphabet alpha, const NRMat<dtype>& wordEmb, bool bFinetune) {

    table.elems = alpha;
    table.nVSize = wordEmb.nrows();
    table.nUNKId = table.elems->from_string(unknownkey);
    table.bFineTune = bFinetune;
    table.nDim = wordEmb.ncols();

    table.E.initial(table.nDim, table.nVSize);
    table.E.val.setZero();

	int dim1 = wordEmb.nrows();
	int dim2 = wordEmb.ncols();
	for (int idx = 0; idx < dim1; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			table.E.val(idx, idy) = wordEmb[idx][idy];
		}
	}

	if (bFinetune){
		for (int idx = 0; idx < table.nVSize; idx++){
			norm2one(table.E.val, idx);
		}
	}


  }

void stat2Alphabet(unordered_map<string, int>& stat, Alphabet& alphabet, const string& label, int wordCutOff) {

	cout << label<<" num: " << stat.size() << endl;
	alphabet.set_fixed_flag(false);
	unordered_map<string, int>::iterator feat_iter;
	for (feat_iter = stat.begin(); feat_iter != stat.end(); feat_iter++) {
		// if not fine tune, add all the words; if fine tune, add the words considering wordCutOff
		// in order to train unknown

			if (feat_iter->second > wordCutOff) {
			  alphabet.from_string(feat_iter->first);
			}

	}
	cout << "alphabet "<< label<<" num: " << alphabet.size() << endl;
	alphabet.set_fixed_flag(true);

}

void randomInitNrmat(NRMat<dtype>& nrmat, Alphabet& alphabet, int embSize, int seed, double initRange) {
	double* emb = new double[alphabet.size()*embSize];
	fox::initArray2((double *)emb, (int)alphabet.size(), embSize, 0.0);

	vector<string> known;
	map<string, int> IDs;
	alphabet2vectormap(alphabet, known, IDs);

	fox::randomInitEmb((double*)emb, embSize, known, unknownkey,
			IDs, false, initRange, seed);

	nrmat.resize(alphabet.size(), embSize);
	array2NRMat((double*) emb, alphabet.size(), embSize, nrmat);

	delete[] emb;
}

string feature_word(const fox::Token& token) {
	//string ret = token.word;
	string ret = normalize_to_lowerwithdigit(token.word);
	//string ret = normalize_to_lowerwithdigit(token.lemma);

	return ret;
}




#endif /* UTILS_H_ */
