//
// Created by arischioppa on 10/7/19.
//

#ifndef CHAT_BOT_PROVISORY_DATA_LOADING_H
#define CHAT_BOT_PROVISORY_DATA_LOADING_H
#include "mxnet-cpp/MxNetCpp.h"

namespace mx = mxnet::cpp;

// don't know the vector type to feed in the embedding, hence IdxType
typedef mx_float IdxType;
// type for vector of word index representing a sentence
typedef std::vector<IdxType> FVec;
// type for a pair of sentences
typedef std::vector<std::pair<FVec, FVec>> PairSenVec;

/**
 * The data iterator; data fits in memory hence loadSentencePairs
 * loads all in memory; note that MxNet execution via Symbol API
 * makes hard to have batches of variable size; this is why we wrap
 * around the batch
 * // todo -> need to make padding fixed too
 */
class BatchIter {
public:
    PairSenVec raw_data;
    BatchIter(mx::Context ctx, int batch_size,
              std::string vocabulary_file, std::string cleaned_conv_file);

    int batch_size() const { return m_batch_size; };
    int vocabulary_size() const { return  m_vocabulary_size; };
    int max_first_sentence_len() const { return m_max_first_sentence_len; };
    int max_second_sentence_len() const { return m_max_second_sentence_len; };

    bool am_exhausted() const { return m_am_exhausted; };

    void reset() {read_pos = 0, m_am_exhausted = false; };

    class ExhaustedBatch : public std::exception {};
    // error if the batch size is too big to get distinct data rows
    class InsufficientData : public  std::exception {};
    struct DataType {
        mx::NDArray input_sentence;
        mx::NDArray input_sequence_lengths;
        mx::NDArray output_sentence;
        mx::NDArray output_sequence_lengths;
        mx::NDArray lagged_output_sentence;
        mx::NDArray output_mask;
    };

    DataType getBatch();
private:
    PairSenVec loadSentencePairs(std::string vocabulary_file, std::string cleaned_conv_file);
    int m_batch_size;
    mx::Context m_ctx;
    bool m_am_exhausted;
    int read_pos = 0;
    int pad_token = -1;
    int eos_token = -1;
    int sos_token = -1;
    int m_vocabulary_size = -1;
    int m_max_first_sentence_len = -1;
    int m_max_second_sentence_len = -1;


};

#endif //CHAT_BOT_PROVISORY_DATA_LOADING_H
