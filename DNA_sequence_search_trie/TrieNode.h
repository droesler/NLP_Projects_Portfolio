
#ifndef TRIENODE_H_
#define TRIENODE_H_

#include <iostream>
#include <sstream>		// integer to hex conversion
#include <map>			// used to store search results
#include <vector>

using namespace std;

//  ##################################################################################################################
//
//  David Roesler (roeslerdavidthomas@gmail.com) "TrieNode.h" header file
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

class TrieNode {

	public:

		//  ===========================================================================================
		//  methods
		//  ===========================================================================================

		TrieNode();
		//  Default constructor. Initializes data members.

		~TrieNode();
		//  Destructor. Calls remove_children to deallocate dynamic memory.

		void remove_children();
		//  Recursively deallocates dynamic memory.

		void add_child(int to_add);
		//  Accepts an integer argument and adds a child node to that index.

		void add_sequence(unsigned int current_index, const string & string_to_add);
		//  Traverses the trie and adds nodes that correspond to the characters in the input string
		//  if those nodes do not already exist.

		int hash_function(const char & to_find);
		//  Accepts a character argument and returns a child node index using a switch.

		void match_sequence(string & dna_content, map<string, vector<string>> & results_map, string & filename);
		//  Searches the DNA data for matches, outputs results to console, and stores
		//  results in the map.

	private:

		//  ===========================================================================================
		//  data fields
		//  ===========================================================================================

		TrieNode* children[5];
		//  Array of 5 TrieNode pointers used to traverse the trie.

		bool is_leaf;
		//  Used to identify leaf nodes.

};

#endif

//  ##################################################################################################################
//
//  End of TrieNode.h file
//
//  ##################################################################################################################

