//
// Created by arischioppa on 7/22/19.
//

#ifndef CHAT_BOT_PROVISORY_DATA_PROCESSING_H
#define CHAT_BOT_PROVISORY_DATA_PROCESSING_H

#include<string>
#include<vector>
#include<map>
#include<iostream>
#include <stdint.h>

/**************************************
 * Represents a MovieLine in movie_lines.txt
 */
class MovieLine {
public:
    std::string m_lineID;
    std::string m_characterID;
    std::string m_movieID;
    std::string m_character_name;
    std::string m_text;

    // default constructor
    MovieLine(std::string lineID,  std::string characterID,
              std::string movieID, std::string character_name,
              std::string text);

    // constructor from a line in movie_lines.txt
    MovieLine(const std::string& s);

};

class MovieConversation {
public:
    std::string m_firstCharID;
    std::string m_secondCharID;
    std::string m_movieID;
    std::vector<std::string> m_utterances;

    // default constructor
    MovieConversation(std::string firstCharID, std::string secondCharID,
                      std::string movieID, std::vector<std::string> utterances);

    // constructor from a line in movie_conversations.txt
    MovieConversation(const std::string& s);
};

class StringTokenizer {
public:
    StringTokenizer(const std::string& s, const char* delim = NULL);

    size_t countTokens( );

    bool hasMoreTokens( );

    void nextToken(std::string& s);

private:
    StringTokenizer( ) {};
    std::string delim_;
    std::string str_;
    int count_;
    int begin_;
    int end_;
};


struct Vocabulary {
    std::map<std::string, std::uint32_t> word_to_index;
    std::map<std::uint32_t, std::string> index_to_word;
    int size = -1;

    Vocabulary(std::istream &is);

    // if lookup fails returns empty
    std::string get_word(std::uint32_t index);

    // if lookup fails returns UINT32_MAX
    std::uint32_t get_index(std::string word);

};

#endif //CHAT_BOT_PROVISORY_DATA_PROCESSING_H
