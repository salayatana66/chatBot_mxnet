#include<fstream>
#include<sstream>
#include<string>
#include<iostream>
#include<vector>
#include<queue>
#include<regex>
#include<stdio.h>
#include "data_processing.h"
#include "data_representation.h"

int main() {
    std::ifstream vocabulary_file("data/vocabulary.txt");
    std::ifstream cleaned_conversations_file("data/cleaned_conversations.txt");

    auto output_binary = std::fopen("data/input_conversations_ubyte", "wb");
    auto vocabulary = Vocabulary(vocabulary_file);

    vocabulary_file.close();

    std::string separator = " +++$+++ ";
    std::string line;
    u_int sep_len = separator.length();

    // stack for unit testing

    std::queue<std::pair<std::string, std::vector<uint>>>
            sentence_tests;

    auto eos_idx = vocabulary.get_index("EOS");
    auto sos_idx = vocabulary.get_index("SOS");
    int iCounter = 0;
    while(std::getline(cleaned_conversations_file, line)) {
        auto pos = line.find(separator, 0);
        auto utt1 = line.substr(0, pos);
        auto utt2 = line.substr(pos + sep_len, std::string::npos);

        // accumulate indices for the first utterance
        std::vector<uint> word_indices;
        StringTokenizer* tokenizer = new StringTokenizer(utt1);

        std::string token_holder;

        word_indices.push_back(sos_idx);
        while(tokenizer->hasMoreTokens()) {
            tokenizer->nextToken(token_holder);
            auto idx = vocabulary.get_index(token_holder);

            // add only tokens that are in the vocabulary
            if(idx != UINT32_MAX) word_indices.push_back(idx);
        }
        word_indices.push_back(eos_idx);

        // push on stack
        sentence_tests.push(std::pair<std::string,
                std::vector<uint>>(utt1, word_indices));



        // add characters for SOS & EOS
        auto sentence_to_serialize = new Sentence(word_indices.size());
        for(int i = 0; i< word_indices.size(); i++)
            sentence_to_serialize->set_content(i, word_indices[i]);

        sentence_to_serialize->serialize(output_binary);


        if(iCounter > 10) break;
        iCounter++;
        delete tokenizer;
    }

    std::fclose(output_binary);

    auto input_binary = std::fopen("data/input_conversations_ubyte", "rb");

    while(!sentence_tests.empty()) {
        auto values = sentence_tests.front();
        sentence_tests.pop();

        std::cout << values.first << " => ";
        for(auto& q: values.second)
            std::cout << q << " -> ";
        std::cout << std::endl;

        auto sentence = new Sentence(input_binary);

        std::cout << "Reread contents ";
        for(int i = 0; i < sentence->length(); ++i)
            std::cout << *(sentence->content()+i) << '\t';
        std::cout << std::endl;
        delete sentence;

    }

    std::fclose(input_binary);
    return 0;
}

