#include "data_processing.h"
#include<vector>
#include<iostream>

MovieLine::MovieLine(std::string lineID, std::string characterID,
                     std::string movieID, std::string character_name,
                     std::string text)
        : m_lineID(std::move(lineID)), m_characterID(std::move(characterID)),
        m_movieID(std::move(movieID)), m_character_name(std::move(character_name)),
        m_text(std::move(text)) {}

MovieLine::MovieLine(const std::string &s) {
    int j = 0;
    std::vector<std::string> v(5);

    // this is the pattern separating fields
    std::string model = " +++$+++ ";

    // there are 5 fields per example
    for(int i = 0; i <5; i++){
        // scan end of next delimiter
        auto pos = s.find(model, j);
        // update the vector
        v[i] = s.substr(j, pos - j);
        // update next search position
        j = pos + model.length();
    }

    m_lineID = v[0];
    m_characterID = v[1];
    m_movieID = v[2];
    m_character_name = v[3];
    m_text = v[4];
}

MovieConversation::MovieConversation(std::string firstCharID, std::string secondCharID, std::string movieID,
                                     std::vector<std::string> utterances)
                                     : m_firstCharID(std::move(firstCharID)), m_secondCharID(std::move(secondCharID)),
                                     m_movieID(std::move(movieID)), m_utterances(std::move(utterances)) {}

MovieConversation::MovieConversation(const std::string &s) {
    int j = 0;
    std::vector<std::string> v(4);

    // this is the pattern separating fields
    std::string model = " +++$+++ ";

    // there are 4 fields per example
    for(int i = 0; i <4; i++){
        // scan end of next delimiter
        auto pos = s.find(model, j);
        // update the vector
        v[i] = s.substr(j, pos - j);
        // update next search position
        j = pos + model.length();
    }

    m_firstCharID = v[0];
    m_secondCharID = v[1];
    m_movieID = v[2];

    // the string with utterances is of form ['uid', 'uid', ..., 'uid'] and we must break it up
    std::string utterances = v[3];

    std::vector<std::string> u;
    int p1, p2 = 0;
    char sep = '\'';

    while(true) {
        // scan for next ' twice and break if fails
        p1 = utterances.find(sep, p2);
        if(p1 == std::string::npos) break;
        p2 = utterances.find(sep, p1 + 1);
        if(p2 == std::string::npos) break;
        // offsets are used to remove the 's
        u.push_back(utterances.substr(p1 + 1, p2 - p1 -1 ));
        p2++;
    }
    m_utterances = std::move(u);
}

StringTokenizer::StringTokenizer(const std::string& s, const char *delim) :
        str_(s), count_(-1), begin_(0), end_(0) {

    if (!delim)
        //default to whitespace
        delim_ = " \f\n\r\t\v";
    else
        delim_ = delim;

    // Point to the first token
    begin_ = str_.find_first_not_of(delim_);
    end_ = str_.find_first_of(delim_, begin_);
}

size_t StringTokenizer::countTokens() {

    if (count_ >= 0) // return if we've already counted
        return (count_);

    std::string::size_type n = 0;
    std::string::size_type i = 0;

    for (;;) {
        // advance to the first token
        if ((i = str_.find_first_not_of(delim_, i)) == std::string::npos)
            break;
        // advance to the next delimiter
        i = str_.find_first_of(delim_, i + 1);
        n++;
        if (i == std::string::npos) break;
    }
    count_ = n;
    return n;

}

bool StringTokenizer::hasMoreTokens() {return(begin_ != end_);}

void StringTokenizer::nextToken(std::string &s) {

    if (begin_ != std::string::npos && end_ != std::string::npos) {
        s = str_.substr(begin_, end_ - begin_);
        begin_ = str_.find_first_not_of(delim_, end_);
        end_ = str_.find_first_of(delim_, begin_);
    } else if (begin_ != std::string::npos &&
               end_ == std::string::npos) {
        s = str_.substr(begin_, str_.length() - begin_);
        begin_ = str_.find_first_not_of(delim_, end_);
    }
}

Vocabulary::Vocabulary(std::istream &is) {
    word_to_index["PAD"] = 0;
    index_to_word[0] = "PAD";
    word_to_index["SOS"] = 1;
    index_to_word[1] = "SOS";
    word_to_index["EOS"] = 2;
    index_to_word[2] = "EOS";

    std::string line;
    std::string separator = " +++$+++ ";
    std::getline(is, line);

    std::uint32_t  vocab_idx = 3;
    while(std::getline(is, line)) {
        // get the word by breaking at the separator
        auto pos = line.find(separator, 0);
        auto word = line.substr(0, pos);

        word_to_index[word] = vocab_idx;
        index_to_word[vocab_idx] = word;
        vocab_idx++;
    }

    // set the vocabulary size (needed later for embeddings)
    size = vocab_idx;
}

std::string Vocabulary::get_word(std::uint32_t index) {
    if(index_to_word.count(index) == 0)
        return std::string();
    else
        return index_to_word[index];

}

std::uint32_t Vocabulary::get_index(std::string word) {
    if(word_to_index.count(word) == 0)
        return UINT32_MAX;
    else
        return word_to_index[word];
}