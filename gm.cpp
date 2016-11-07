/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#include <vector>
#include "utils.h"
#include <iostream>
#include "Argument_helper.h"
#include "Options.h"
#include "Tool.h"


#include "NNgm.h"
#include "NNgm2.h"
#include "NNgm3.h"


using namespace std;


int main(int argc, char **argv)
{



	string optionFile;
	string trainFile;
	string testFile;
	string outputFile;
	string trainAnswerFile;
	string testAnswerFile;
	vector<string> kbFiles;



	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("output", "", "", "", outputFile);
	ah.new_named_string("trainAnswer", "", "", "", trainAnswerFile);
	ah.new_named_string("testAnswer", "", "", "", testAnswerFile);
	ah.new_named_string_vector("kb", "", "", "", kbFiles);


	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"test file: "<<testFile<<endl;

	cout<<"trainnlp file: "<<trainAnswerFile<<endl;
	cout<<"testnlp file: "<<testAnswerFile<<endl;

	cout<<"knowledge base: ";
	for(int i=0;i<kbFiles.size();i++) {
		cout<<kbFiles[i]<<" ";
	}
	cout<<endl;


	Options options;
	options.load(optionFile);

	if(!outputFile.empty())
		options.output = outputFile;

	options.showOptions();

	Tool tool(options);


	//NNgm nn(options);
	//NNgm2 nn(options);
	NNgm3 nn(options);

	nn.trainAndTest(trainFile, testFile, tool,
			trainAnswerFile, testAnswerFile, kbFiles);




    return 0;

}

