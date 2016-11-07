/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "MyLib.h"

using namespace std;
struct Feature {
public:
	vector<string> words; // actually only words[0] is used
	vector<string> types; // other kinds of features, such as POS

public:
	Feature() {
	}

	//virtual ~Feature() {
	//
	//}

	void clear() {
		words.clear();
		types.clear();
	}
};

class Example { // one example corresponds to one sentence

public:
	vector<vector<dtype> > m_labels;

	vector<Feature> m_features; // one feature corresponds to one word

public:
	Example()
	{

	}
	virtual ~Example()
	{

	}

	void clear()
	{
		m_features.clear();
		clearVec(m_labels);

	}


};


#endif /* SRC_EXAMPLE_H_ */
