
#include "TrieNode.h"
using namespace std;

//  ##################################################################################################################
//
//  David Roesler (roeslerdavidthomas@gmail.com) "TrieNode.cpp" TrieNode class implementation file
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
//  TrieNode::TrieNode:
//  Default constructor. Initializes data members.
//  ===========================================================================================

TrieNode::TrieNode() {
	children[0] = NULL;	// A
	children[1] = NULL;	// C
	children[2] = NULL;	// G
	children[3] = NULL;	// T
	children[4] = NULL;	// N and other
	is_leaf = true;
}

//  ===========================================================================================
//  TrieNode::~TrieNode:
//  Destructor. Calls remove_children to deallocate dynamic memory.
//  ===========================================================================================

TrieNode::~TrieNode() {
	this->remove_children();
}

//  ===========================================================================================
//  TrieNode::remove_children:
//  Recursively deallocates dynamic memory.
//  ===========================================================================================

void TrieNode::remove_children() {
	for (int i = 0; i < 4; ++i) {// index 4 will always be null, so it is not examined
		if (children[i] != NULL) {
			children[i]->remove_children();	// recursive call
			delete children[i];
			children[i] = NULL;
		}
	}
}

//  ===========================================================================================
//  TrieNode::hash_function:
//  Accepts a character argument and returns a child node index using a switch.
//  ===========================================================================================

int TrieNode::hash_function(const char &to_find) {
	switch (to_find) {
	case 'A':
		return 0;
	case 'C':
		return 1;
	case 'G':
		return 2;
	case 'T':
		return 3;
	default:
		return 4;
	}
}

//  ===========================================================================================
//  TrieNode::add_child:
//  Accepts an integer argument and adds a child node to that index.
//  Marks the node as "not a leaf" after a child is added.
//  ===========================================================================================

void TrieNode::add_child(int to_add) {
	children[to_add] = new TrieNode();
	is_leaf = false;
}

//  ===========================================================================================
//  TrieNode::add_sequence:
//  Accepts an integer and a string as arguments.
//  Traverses the trie and adds nodes that correspond to the characters in the input string
//  if those nodes do not already exist.
//  ===========================================================================================

void TrieNode::add_sequence(unsigned int current_index, const string &string_to_add) {
	if (current_index == string_to_add.length())
		return;
	char current_char = string_to_add[current_index]; // get char from string
	int index_to_add = hash_function(current_char);	// get index from hash function
	if (index_to_add < 4 && this->children[index_to_add] == NULL) // add node if is null
		this->add_child(index_to_add);
	this->children[index_to_add]->add_sequence(current_index + 1,
			string_to_add); // recursive call
}

//  ===========================================================================================
//  TrieNode::match_sequence:
//  Accepts a string (DNA data), a map, and another string (file name) as arguments.
//  Searches the DNA data for matches, outputs results to console, and stores
//  results in the map.
//  ===========================================================================================

void TrieNode::match_sequence(string & dna_content, map<string, vector<string>> & results_map,
		string & filename) {

	char *current_char;						// char pointer to "j" index element
	char *start_char;						// char pointer to "i" index element
	TrieNode *current_node = NULL;			// used to traverse trie nodes
	TrieNode *next_node = NULL;				// used to check next node for NULL
	string hex_result, match_string;
	stringstream stream;							// used for hex conversion

	cout << filename << "\n";
	for (unsigned int i = 0; i < dna_content.length(); ++i) {// for each character
		current_char = start_char = &(dna_content[i]);		// reset pointers
		if (*current_char != 'N') {
			current_node = this;

			// find next node using TrieNode class hash function
			next_node = current_node->children[current_node->hash_function(
					*current_char)];
			while (next_node != NULL) {	// traverse nodes until cannot proceed
				current_node = next_node;
				current_char += 1;
				next_node = current_node->children[current_node->hash_function(
						*current_char)];
			}
			if (current_node->is_leaf) {// check for leaf using TrieNode data member
				stream << std::hex << i;
				hex_result = stream.str();	// get hex value for match index
				stream.str(string());		// flush stringstream
				cout << "\t" << hex_result;
				match_string = dna_content.substr(i, current_char - start_char);
				cout << "\t" << match_string << "\n";
				results_map[match_string].push_back(hex_result);// store hex in map
				results_map[match_string].push_back(filename);	// store filename in map
			}
		}
	}
}

//  ##################################################################################################################
//
//  End of TrieNode.cpp file
//
//  ##################################################################################################################

