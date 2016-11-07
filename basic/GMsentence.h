/*
 * GMsentence.h
 *
 *  Created on: Oct 23, 2016
 *      Author: fox
 */

#ifndef BASIC_GMSENTENCE_H_
#define BASIC_GMSENTENCE_H_

#include <string>
#include <vector>
#include "Token.h"
#include "Entity.h"

using namespace std;

class GMsentence {
public:
	string id; // sentence id
	string text; // sentence text
	vector <fox::Token> tokens; // the text in this sentence and has been tokenized
	vector <Entity> entities;

	GMsentence() {

	}
};

#endif /* BASIC_GMSENTENCE_H_ */
