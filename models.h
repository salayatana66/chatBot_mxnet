//
// Created by arischioppa on 10/7/19.
//

#ifndef CHAT_BOT_PROVISORY_MODELS_H
#define CHAT_BOT_PROVISORY_MODELS_H

#include <string>
#include "mxnet-cpp/MxNetCpp.h"

namespace mx = mxnet::cpp;

mx::Symbol createEncoder(std::string prefix, uint32_t state_size,
                   uint32_t num_layers,
                   mx::Symbol& input_embedding,
                   mx::Symbol& sequence_length,
                   mx::Symbol& initial_state,
                   mx::Symbol& rnn_params,
                   mx_float dropout =0
);

mx::Symbol createDecoder(std::string prefix, uint32_t state_size,
                         uint32_t num_layers,
                         mx::Symbol& input_embedding,
                         mx::Symbol& encoder_inner_state,
                         mx::Symbol& sequence_length,
                         mx::Symbol& rnn_params,
                         mx_float dropout = 0
);

enum class AttentionMode {
    kDot = 0,
    kGeneral = 1,
    kConcat = 2
};

std::ostream& operator <<(std::ostream& out, const AttentionMode& attentionMode);

std::pair<std::map<std::string, mx::Symbol>,
        mx::Symbol> createAttention(std::string prefix,
                int hidden_dimension,
                int encoder_max_sequence_length,
                int decoder_max_sequence_length,
                int batch_size,
                mx::Symbol& encoder_output,
                mx::Symbol& decoder_output,
                std::map<std::string, mx::Symbol>& params,
                AttentionMode mode=AttentionMode::kDot);

#endif //CHAT_BOT_PROVISORY_MODELS_H
