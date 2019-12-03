//
// Created by arischioppa on 7/25/19.
//

#include "data_processing.h"
#include<fstream>
#include <iostream>
#include <ctime>
#include <map>
#include <algorithm>

int main() {
    // input files
    std::ifstream convFile("data/movie_conversations.txt");
    std::ifstream lineFile("data/movie_lines.txt");

    // output file
    std::ofstream outFile;
    outFile.open("data/formatted_conversations.txt");

    // map to store lines
    std::map<std::string, MovieLine*> movieLines;

    auto b_movie_line_time = std::clock();
    std::cout << "Loading movie lines" << std::endl;

    std::string line;
    while(std::getline(lineFile, line)) {
        auto *ml = new MovieLine(line);
        movieLines[ml->m_lineID] = ml;
    }
    auto e_movie_line_time = std::clock();

    std::cout << "Finished loaded movie lines: " <<
    double(e_movie_line_time - b_movie_line_time) / CLOCKS_PER_SEC <<
    std::endl;

    std::cout << "Loading & Writing conversations (pairs of lines)" <<
    std::endl;

    auto b_movie_conv_time = std::clock();

    // Writing pairs of conversations
    while(std::getline(convFile, line)) {
        // get a conversations
        auto *mc = new MovieConversation(line);
        int utt_size =  mc->m_utterances.size();

        // loop over the list of lines
        std::string first_line;
        std::string second_line;
        for(int i = 0; i < utt_size - 1; i++) {
            /*
            Note the pairs are linearly arranged in the vector

             */
            first_line = movieLines[mc->m_utterances[i]]->m_text;
            second_line = movieLines[mc->m_utterances[i+1]]->m_text;

            // remove inner tabs & quotes
            outFile << first_line << " +++$+++ " <<
            second_line << "\n";
        }
        delete mc;
    }

    auto e_movie_conv_time = std::clock();
    std::cout << "Finished loading & writing conversation pairs: " <<
              double(e_movie_conv_time - b_movie_conv_time) / CLOCKS_PER_SEC <<
              std::endl;

    outFile.close();
    convFile.close();
    lineFile.close();
    // TODO: if add more stuff may free the map with lines
    return 0;
}

