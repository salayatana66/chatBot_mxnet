//
// Created by arischioppa on 7/28/19.
//
#include<fstream>
#include<iostream>
#include<vector>
#include<regex>
#include<chrono>
#include "data_processing.h"

// maximal length allowed for a sentence
int MAX_LENGTH = 15;

// Minimum word count threshold for trimming
int MIN_COUNT = 3;

int main() {

    // input file
    std::ifstream infile("data/formatted_conversations.txt");
    std::string line;

    /* output files
     * -- conversations with clean words
     * -- output vocabulary
     */
    std::ofstream cleanedFile;
    cleanedFile.open("data/cleaned_conversations.txt");
    std::ofstream vocabFile;
    vocabFile.open("data/vocabulary.txt");

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Creating Vocabulary\n" <<
    "and cleaning words characters and discarding conversations exceeding\n" <<
    "length: " << MAX_LENGTH << std::endl;

    start = std::chrono::steady_clock::now();
    // vocabulary to hold word counts & word identifier
    std::map<std::string, uint> vocabulary;

    /*********************************************
     * Processing the formatted conversations to build
     * The vocabulary
     ************************************************/
    std::string separator = " +++$+++ ";
    u_int sep_len = separator.length();

    // match non-letter characters
    std::regex non_letter("[^a-zA-Z.!?]+");
    // match consecutive spaces
    std::regex cons_spaces("\\s+");
    // match one or more !, ?, . and replace with a space
    std::regex exclamation_marks("!+");
    std::regex question_marks("\\?+");
    std::regex the_dots("\\.+");

    while(std::getline(infile, line)) {

        auto pos = line.find(separator, 0);
        auto utt1 = line.substr(0, pos);
        auto utt2 = line.substr(pos + sep_len, std::string::npos);

        utt1 = std::regex_replace(utt1, non_letter, " ");
        utt1 = std::regex_replace(utt1, exclamation_marks, " ! ");
        utt1 = std::regex_replace(utt1, question_marks, " ? ");
        utt1 = std::regex_replace(utt1, the_dots, " . ");
        // string to lower case the functional way
        std::transform(utt1.begin(), utt1.end(),
                utt1.begin(), [](unsigned char c){ return std::tolower(c); });
        // remove consecutive spaces
        utt1 = std::regex_replace(utt1, cons_spaces, " ");

        utt2 = std::regex_replace(utt2, non_letter, " ");
        utt2 = std::regex_replace(utt2, exclamation_marks, " ! ");
        utt2 = std::regex_replace(utt2, question_marks, " ? ");
        utt2 = std::regex_replace(utt2, the_dots, " . ");

        // string to lower case the functional way
        std::transform(utt2.begin(), utt2.end(), utt2.begin(),
                [](unsigned char c){ return std::tolower(c); });
        // remove consecutive spaces
        utt2 = std::regex_replace(utt2, cons_spaces, " ");

        StringTokenizer* tokenizer1 = new StringTokenizer(utt1);
        StringTokenizer* tokenizer2 = new StringTokenizer(utt2);

        // check whether both sentences are short enough
        // to be written

        if((tokenizer1 -> countTokens() < MAX_LENGTH) &
                (tokenizer2 -> countTokens() < MAX_LENGTH)) {
            cleanedFile << utt1 << separator << utt2 << '\n';
        }

        std::string token_holder;

        while(tokenizer1 -> hasMoreTokens()) {
            tokenizer1->nextToken(token_holder);

            auto is_in_vocabulary = vocabulary.find(token_holder);

            if(is_in_vocabulary == vocabulary.end())
                vocabulary[token_holder] = 1;
            else
                vocabulary[token_holder]++;

        }

        while(tokenizer2 -> hasMoreTokens()) {
            tokenizer2->nextToken(token_holder);

            auto is_in_vocabulary = vocabulary.find(token_holder);

            if(is_in_vocabulary == vocabulary.end())
                vocabulary[token_holder] = 1;
            else
                vocabulary[token_holder]++;

        }

        delete tokenizer1;
        delete tokenizer2;

    }

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;

    std::cout << "Finished writing vocabulary:" <<
    elapsed_seconds.count() << std::endl;

    std::cout << "Exporting vocabulary\n" <<
    "Word occurring less than " << MIN_COUNT <<
    " times are removed\n";

    start = std::chrono::steady_clock::now();
    // write the vocabulary
    vocabFile << "Word\tCount\n";

    for(auto& m : vocabulary) {
        if(m.second >= MIN_COUNT)
            vocabFile << m.first << separator << m.second << '\n';
    }

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Finished writing vocabulary: " <<
    elapsed_seconds.count() << std::endl;

    cleanedFile.close();
    vocabFile.close();
    infile.close();
    return 0;
}
