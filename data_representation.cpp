//
// Created by arischioppa on 8/18/19.
//

#include "data_representation.h"


Sentence::Sentence(int length): m_length(length){
    m_content = new uint[length];
}

Sentence::Sentence(std::FILE *infile) {
    m_length = 0;
    // read actual length
    std::fread(&m_length, sizeof(int), 1, infile);

    // allocate memory for content and read it
    m_content = new uint[m_length];
    std::fread(m_content, sizeof(uint), m_length, infile);
}

void Sentence::set_content(const int index, uint &value) {
    *(m_content + index) = value;
}

void Sentence::serialize(std::FILE *outfile) {

    std::fwrite(&m_length, sizeof(int), 1, outfile);
    std::fwrite(m_content, sizeof(uint), m_length, outfile);
}


/*InputBatch::InputBatch(int batch_size, int max_length)
:m_batch_size(batch_size), m_max_length(max_length)
{
    m_values = new std::uint32_t[batch_size * max_length];
}

void InputBatch::write(std::FILE *outfile) const {
    std::fwrite(&m_batch_size, sizeof(int), 1, outfile);
    std::fwrite(&m_max_length, sizeof(int), 1, outfile);
    std::fwrite(m_values, sizeof(std::uint32_t), m_batch_size*m_max_length,
            outfile);
}

InputBatch::InputBatch(std::FILE *infile) {
    std::fread(&m_batch_size, sizeof(int), 1, infile);
    std::fread(&m_max_length, sizeof(int), 1, infile);

    m_values = new std::uint32_t[m_batch_size * m_max_length];
    std::fread(m_values, sizeof(std::uint32_t), m_batch_size * m_max_length,
            infile);

}
*/