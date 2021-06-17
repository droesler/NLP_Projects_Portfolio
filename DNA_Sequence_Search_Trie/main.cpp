
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include "TrieNode.h"		// custom Trienode class
#include <regex>		// used to identify *.dna files
#include <dirent.h>		// used to read directory contents
using namespace std;


//  ##################################################################################################################
//
//  David Roesler (roeslerdavidthomas@gmail.com) "main.cpp" Main program file
//
//  ##################################################################################################################
//
//  DNA Sequence Trie Search
//
//  This program searches for 4,965 target DNA sequences in a set of 24 files that map the chromosomes of
//  the human genome. The program contains a TrieNode class which builds a trie structure from the list of target
//  sequences and allows for relatively quick simultaneous searches of all target sequences in the large chromosome
//  data files. Each node in the trie structure contains an array of five TrieNode pointers corresponding to the
//  five types of DNA input: 'A', 'C', 'G', 'T', and 'N'. The class uses a C++ switch statement as a simple hash
//  function that returns the index number for child nodes in the trie structure. Each node also contains a boolean
//  value (is_leaf) that corresponds to its node status. This marker prevents the need to check all of a nodes'
//  child pointers for 'null' in order to determine whether a node is a terminal/leaf node in the trie. Results
//  are output to the console and also output to an "extra-credit" file in a second format.
//
//  ##################################################################################################################


//  ===========================================================================================
//  prototypes
//  ===========================================================================================


void build_trie_from_file(TrieNode* & root);
void get_dna_files(vector<string> & file_list);
void search_trie(TrieNode* & root, vector<string> & file_list);
void write_extra_credit_file(map<string, vector<string>> & results_map);
void dna_file_to_string(string & dna_content, vector<string>::iterator & file_itr);


//  ===========================================================================================
//  main
//  ===========================================================================================


int main() {
	TrieNode *root = new TrieNode;		// root node of trie
	vector<string> file_list;

	build_trie_from_file(root);		// builds trie from "targets" file
	get_dna_files(file_list);		// fills a vector with names of *.dna files
	search_trie(root, file_list); 		// search the trie and output results
	delete root;				// deallocate dynamic memory
	
	return 0;
}


//  ===========================================================================================
//  build_trie_from_file:
//  Reads targets file and builds trie structure.
//  ===========================================================================================


void build_trie_from_file(TrieNode* & root){
	ifstream target_file_in;
	string string_to_add;

	target_file_in.open("targets");	// open target file
	if (target_file_in) {
		target_file_in >> string_to_add;			// read first line
		while (target_file_in && !target_file_in.eof()) {
			target_file_in.ignore(100, '\n');
			root->add_sequence(0, string_to_add);		// add sequence to trie
			target_file_in >> string_to_add;
		}
	}
	target_file_in.close();
}


//  ===========================================================================================
//  get_dna_files:
//  Packages .dna file names into a string vector using "dirent.h".
//  ===========================================================================================


void get_dna_files(vector<string> & file_list){
	DIR *dir;
	struct dirent *pdir;
	regex dna(".*\\.dna");
	
	dir = opendir("/hg19-GRCh37/");
	while((pdir = readdir(dir)) != NULL){
		string temp = pdir->d_name;			// read directory item
		if (regex_match(temp,dna))			// match .dna extension
			file_list.push_back(temp);		// add to vector
	}
	closedir(dir);
	sort(file_list.begin(), file_list.end());
}


//  ===========================================================================================
//  search_trie:
//  Finds and outputs target matches in .dna files.
//  ===========================================================================================


void search_trie(TrieNode* & root, vector<string> & file_list){
	vector<string>::iterator file_itr;			// iterates over .dna files
	string dna_content;					// stores .dna file contents
	map<string, vector<string>> results_map;		// stores search results

	// for each .dna file
	for(file_itr=file_list.begin();file_itr != file_list.end();++file_itr){

		// convert .dna file to formatted string
		dna_file_to_string(dna_content, file_itr);

		// get matches from TrieNode class method
		root->match_sequence(dna_content, results_map, *file_itr);
	}
	write_extra_credit_file(results_map);	// write extra-credit file
}


//  ===========================================================================================
//  dna_file_to_string: (called by search_trie)
//  Converts .dna file contents to uppercase string.
//  ===========================================================================================


void dna_file_to_string(string & dna_content, vector<string>::iterator & file_itr){
	ifstream dna_file_in;
	dna_file_in.open("/hg19-GRCh37/" + *file_itr);
	if (dna_file_in) {
		dna_file_in >> dna_content;			// read into string
		for (unsigned int j = 0; j < dna_content.length(); ++j)
			dna_content.at(j) &= ~0x20;		// bit shift all chars to upper case
	}
	dna_file_in.close();
}


//  ===========================================================================================
//  write_extra_credit_file: (called by search_trie)
//  Outputs target matches followed by their file occurrences and corresponding hex offsets.
//  ===========================================================================================


void write_extra_credit_file(map<string, vector<string>> & results_map){
	ofstream file_out;
	file_out.open("output_ver2");
	if (file_out) {
		for(map<string, vector<string>>::iterator ii = results_map.begin();ii!=results_map.end();++ii){
			file_out << ii->first << "\n";					// output target DNA sequence
			vector<string> temp_vect = ii->second;			// temp bucket for hex and filename results
			for(unsigned k=0;k<temp_vect.size();++k){
				file_out << "\t" << temp_vect[k];			// output hex offset
				++k;
				file_out << "\t" << temp_vect[k] << "\n";	// output file name
			}
		}
	}
	file_out.close();
}


//  ##################################################################################################################
//
//  End of main.cpp file
//
//  ##################################################################################################################

