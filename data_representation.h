//
// Created by arischioppa on 8/18/19.
//

#ifndef CHAT_BOT_PROVISORY_DATA_REPRESENTATION_H
#define CHAT_BOT_PROVISORY_DATA_REPRESENTATION_H

#include <stdint.h>
#include <iostream>
#include <stdio.h>

/****
 *
 * Represents a sentence for serialization to a file
 */

class Sentence {
private:
    int m_length;
    u_int* m_content;

public:
    Sentence(int length);
    Sentence(std::FILE* infile);

    int length() const {return m_length;};
    const u_int* content() const {return m_content;};

    void set_content(const int index, uint& value);

    void serialize(std::FILE* outfile);

};

/*class InputBatch {
public:
    InputBatch(int batch_size, int max_length);
    InputBatch(std::FILE* infile);

    int batch_size() const {return m_batch_size;};
    int max_length() const {return m_max_length;};
    const std::uint32_t * values() const {return m_values;};

    void write(std::FILE* outfile) const;

private:
    int m_batch_size;
    int m_max_length;
    std::uint32_t* m_values;
};
 */
#endif //CHAT_BOT_PROVISORY_DATA_REPRESENTATION_H
