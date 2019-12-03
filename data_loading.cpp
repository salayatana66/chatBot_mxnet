//
// Created by arischioppa on 10/7/19.
//
#include<iostream>
#include<fstream>

#include "data_loading.h"
#include "data_processing.h"

/** If it reaches the end of the batch it wraps around to get to
 * the right batch size -> Symbol API wants fixed dims
 */
BatchIter::DataType BatchIter::getBatch() {
    if(m_am_exhausted) throw BatchIter::ExhaustedBatch();

    int start = read_pos;
    int end = read_pos + m_batch_size;
    if(end >= raw_data.size()) end = raw_data.size();
    // size of the batch
    int width = m_batch_size;

    /** all batches on each kind of sentence, first vs second,
     *  need to have the same time size
     */
    int slen_first = m_max_first_sentence_len,
            slen_second = m_max_second_sentence_len;

    // allocate memory for data
    // the p_ pointer is used to fill the data array
    // with data from the batch
    IdxType *data_first = new IdxType[slen_first * width],
            *p_data_first = data_first;

    IdxType *seq_len_first = new IdxType[width],
            *p_seq_len_first = seq_len_first;

    IdxType *data_second = new IdxType[slen_second * width],
            *p_data_second = data_second;

    IdxType *seq_len_second = new IdxType[width],
            *p_seq_len_second = seq_len_second;

    // we use this to lag the second sentence
    IdxType  *data_lagged_second = new IdxType[slen_second * width],
    *p_data_lagged_second = data_lagged_second;

    // we use to keep track of the lagging and have a mask of 1.0s
    // in valid entries and 0.0s elsewhere

    IdxType *data_output_mask = new IdxType[slen_second * width],
    *p_data_output_mask= data_output_mask;


    int j = 0, curr_pos = start;
    while(j < width) {
        auto first_len = raw_data[curr_pos].first.size();
        auto second_len = raw_data[curr_pos].second.size();

        // sentences are initialized with the padding element and in case
        // overwritten
        // some dirty tricks might make this faster with memset, and
        // avoiding going multiple times through the same array
        // but the speed up is not huge.

        for(int init_first = 0; init_first < slen_second; init_first++)
            *(p_data_first+init_first) = static_cast<IdxType>(pad_token);

        for(int init_second = 0; init_second < slen_second; init_second++) {
                *(p_data_second+init_second) = static_cast<IdxType>(pad_token);
                *(p_data_lagged_second+init_second) = static_cast<IdxType>(pad_token);
                *(p_data_output_mask+init_second) = 0.0f;
        }


        // for the input & output sentence it is obvious
        memcpy(p_data_first, raw_data[curr_pos].first.data(), first_len*sizeof(IdxType));
        memcpy(p_data_second, raw_data[curr_pos].second.data(), second_len*sizeof(IdxType));
        // fill in the mask using a for loop
        for(int out_idx = 0; out_idx < second_len; out_idx++)
            *(p_data_output_mask+out_idx) = static_cast<IdxType>(1.0f);

        // for the lagged output sentence insert first an eos
        *p_data_lagged_second = static_cast<IdxType>(eos_token);
        p_data_lagged_second++;
        memcpy(p_data_lagged_second, raw_data[curr_pos].second.data(), (second_len-1)*sizeof(IdxType));

        p_data_first += slen_first;
        *p_seq_len_first = first_len;
        p_seq_len_first++;

        p_data_second += slen_second;
        p_data_output_mask += slen_second;
        p_data_lagged_second += (slen_second-1);
        *p_seq_len_second = second_len;
        p_seq_len_second++;

        curr_pos++;
        j++;

        // reset if reached the end and the batch is not complete
        if(curr_pos >= end && j < width) curr_pos = 0;

    }

    read_pos += width;
    // on end decided if exhausted
    if(end == raw_data.size()) m_am_exhausted = true;
    mx::NDArray array_input(mx::Shape(width, slen_first), m_ctx, false);
    array_input.SyncCopyFromCPU(data_first, width * slen_first);
    mx::NDArray array_input_len(mx::Shape(width), m_ctx, false);
    array_input_len.SyncCopyFromCPU(seq_len_first, width);
    mx::NDArray array_output(mx::Shape(width, slen_second), m_ctx, false);
    array_output.SyncCopyFromCPU(data_second, width * slen_second);
    mx::NDArray array_output_mask(mx::Shape(width, slen_second), m_ctx, false);
    array_output_mask.SyncCopyFromCPU(data_output_mask, width*slen_second);
    mx::NDArray array_output_len(mx::Shape(width), m_ctx, false);
    array_output_len.SyncCopyFromCPU(seq_len_second, width);
    mx::NDArray array_lagged_output(mx::Shape(width, slen_second), m_ctx, false);
    array_lagged_output.SyncCopyFromCPU(data_lagged_second, width*slen_second);

    return DataType{.input_sentence = array_input,
            .input_sequence_lengths = array_input_len,
            .output_sentence = array_output,
            .output_sequence_lengths = array_output_len,
            .lagged_output_sentence = array_lagged_output,
            .output_mask = array_output_mask};

}

BatchIter::BatchIter(mx::Context ctx, int batch_size, std::string vocabulary_file, std::string cleaned_conv_file)
        :m_batch_size(batch_size), m_ctx(ctx)
{
    raw_data = loadSentencePairs(vocabulary_file, cleaned_conv_file);
    m_am_exhausted = raw_data.empty();
    if(m_batch_size > raw_data.size()) throw InsufficientData();

    /** determine max length of sentence on left and right
     *  padding needs to be done statically in the Symbol API
     */
    int slen_first = 0, slen_second = 0;
    for(auto q : raw_data) {
        if(q.first.size() > slen_first)
            slen_first = q.first.size();

        if(q.second.size() > slen_second)
            slen_second = q.second.size();
    }
    m_max_second_sentence_len = slen_second;
    m_max_first_sentence_len = slen_first;

}

/**
 * Loads sentence pairs from cleaned sentences and a vocabulary file
 * @param vocabulary_file
 * @param cleaned_conv_file
 * @return
 */
PairSenVec BatchIter::loadSentencePairs(std::string vocabulary_file, std::string cleaned_conv_file) {

    // create streams
    std::ifstream vocabulary_stream(vocabulary_file);
    std::ifstream cleaned_conv_stream(cleaned_conv_file);

    auto vocabulary = Vocabulary(vocabulary_stream);
    m_vocabulary_size = vocabulary.size;
    vocabulary_stream.close();


    std::string separator = " +++$+++ ";
    std::string line;
    u_int sep_len = separator.length();

    auto eos_idx = vocabulary.get_index("EOS");
    eos_token = eos_idx;
    auto sos_idx = vocabulary.get_index("SOS");
    sos_token = sos_idx;

    // initialize the pad_token for the object
    pad_token = vocabulary.get_index("PAD");

    PairSenVec out;

    while(std::getline(cleaned_conv_stream, line)) {

        auto pos = line.find(separator, 0);
        auto utt1 = line.substr(0, pos);
        auto utt2 = line.substr(pos + sep_len, std::string::npos);

        // accumulate indices for the first utterance
        std::vector<uint> word_indices;
        auto first_tokenizer = new StringTokenizer(utt1);
        auto second_tokenizer = new StringTokenizer(utt2);

        FVec first, second;

        first.push_back((IdxType)sos_idx);
        second.push_back((IdxType)sos_idx);

        std::string token_holder;

        while(first_tokenizer->hasMoreTokens()) {
            first_tokenizer->nextToken(token_holder);
            auto idx = vocabulary.get_index(token_holder);

            // add only tokens that are in the vocabulary
            if(idx != UINT32_MAX) first.push_back((IdxType)idx);
        }

        while(second_tokenizer->hasMoreTokens()) {
            second_tokenizer->nextToken(token_holder);
            auto idx = vocabulary.get_index(token_holder);

            // add only tokens that are in the vocabulary
            if(idx != UINT32_MAX) second.push_back(idx);
        }

        first.push_back((IdxType)eos_idx);
        second.push_back((IdxType)eos_idx);

        delete first_tokenizer, delete second_tokenizer;
        out.push_back(std::make_pair(first, second));

    }

    cleaned_conv_stream.close();

    return out;

}

