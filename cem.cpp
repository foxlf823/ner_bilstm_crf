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


#include "NNcem.h"
#include "NNcem2.h"
#include "NNcem3.h"


using namespace std;


int main(int argc, char **argv)
{



	string optionFile;
	string trainFile;
	string devFile;
	string testFile;
	string outputFile;
	string trainNlpFile;
	string devNlpFile;
	string testNlpFile;
	vector<string> kbFiles;



	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("dev", "", "", "", devFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("output", "", "", "", outputFile);
	ah.new_named_string("trainnlp", "", "", "", trainNlpFile);
	ah.new_named_string("devnlp", "", "", "", devNlpFile);
	ah.new_named_string("testnlp", "", "", "", testNlpFile);
	ah.new_named_string_vector("kb", "", "", "", kbFiles);


	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"dev file: "<<devFile<<endl;
	cout<<"test file: "<<testFile<<endl;

	cout<<"trainnlp file: "<<trainNlpFile<<endl;
	cout<<"devnlp file: "<<devNlpFile<<endl;
	cout<<"testnlp file: "<<testNlpFile<<endl;

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


	NNcem nn(options);
	//NNcem2 nn(options);
	nn.trainAndTest(trainFile, devFile, testFile, tool,
				trainNlpFile, devNlpFile, testNlpFile);

/*	NNcem3 nn(options);

	nn.trainAndTest(trainFile, devFile, testFile, tool,
			trainNlpFile, devNlpFile, testNlpFile, kbFiles);*/




    return 0;

}

