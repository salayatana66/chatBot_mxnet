//
// Created by arischioppa on 10/7/19.
//

#include "models.h"

std::ostream& operator <<(std::ostream& out, const AttentionMode& attentionMode) {
    switch(attentionMode) {
        case AttentionMode::kDot:
            out << "Dot";
            break;
        case AttentionMode::kConcat:
            out << "Concat";
            break;
        case AttentionMode::kGeneral:
            out << "General";
            break;
    }
    return out;
}

mx::Symbol createEncoder(std::string prefix, uint32_t state_size,
                   uint32_t num_layers,
                   mx::Symbol& input_embedding,
                   mx::Symbol& sequence_length,
                   mx::Symbol& initial_state,
                   mx::Symbol& rnn_params,
                   mx_float dropout
) {

    // correct way to make this run
    // was discovered with the Python trick
    auto myRNN = RNN(prefix, input_embedding, rnn_params,
            initial_state, mx::Symbol(),
            sequence_length, state_size, num_layers,
            mx::RNNMode::kGru, true, dropout,
            true, dmlc::optional<int>(),
            dmlc::optional<double>(),
            dmlc::optional<double>(),
            false, true);

    return myRNN;
}

mx::Symbol createDecoder(std::string prefix, uint32_t state_size,
                         uint32_t num_layers,
                         mx::Symbol& input_embedding,
                         mx::Symbol& encoder_inner_state,
                         mx::Symbol& sequence_length,
                         mx::Symbol& rnn_params,
                         mx_float dropout) {


     // correct way to make this run
    // note how the encoder inner state is used
    // to feed the initial state of the GRU
    // as this is a prediction task step by step
    // we cannot use a bidirectional network
    auto myRNN = RNN(prefix, input_embedding, rnn_params,
                     encoder_inner_state, mx::Symbol(),
                     sequence_length, state_size, num_layers,
                     mx::RNNMode::kGru, false, dropout,
                     true, dmlc::optional<int>(),
                     dmlc::optional<double>(),
                     dmlc::optional<double>(),
                     false, true);

    return myRNN;
}

std::pair<std::map<std::string, mx::Symbol>,
        mx::Symbol> createAttention(std::string prefix,
                                    int hidden_dimension,
                                    int encoder_max_sequence_length,
                                    int decoder_max_sequence_length,
                                    int batch_size,
                                    mx::Symbol& encoder_output,
                                    mx::Symbol& decoder_output,
                                    std::map<std::string, mx::Symbol>& params,
                                    AttentionMode mode) {
    int L1 = encoder_max_sequence_length;
    int L2 = decoder_max_sequence_length;
    int H = hidden_dimension;
    int B = batch_size;

    // encoder_output: (L1, B, H), B -> Batch, H -> hidden dimension
    // decoder_output: (L2, B, H), B -> Batch, H -> hidden dimension

    // this will hold the scores that are computed depending on the mode
    mx::Symbol scores;

    if(mode == AttentionMode::kDot) {
        // encoder_output: (L1, B, H) -> (B, H, L1)
        // decoder_output: (L2, B, H) -> (B, L2, H)
        // then batch dot product to get (B, L2, L1)
        auto enc_trans = mx::transpose(prefix+"_encoder_transposed",
                encoder_output, mx::Shape(1, 2, 0));
        auto dec_trans = mx::transpose(prefix+"_decoder_transposed",
                                       decoder_output, mx::Shape(1, 0, 2));

        scores = mx::batch_dot(prefix+"_dot_product", dec_trans, enc_trans);
    } else if (mode == AttentionMode::kConcat) {
        // transformation of encoder_output: (L1, B, H) -> (B, 1, L1, H)
        // -> repeat (B, L2, L1, H)
        auto enc_trans = mx::repeat(prefix+"_encoder_transformed",
                mx::expand_dims(
                mx::transpose(encoder_output, mx::Shape(1, 0, 2)),
                1), decoder_max_sequence_length, dmlc::optional<int>(1));
        // transformation of decoder output: (L2, B, H) -> (B, L2, 1, H)
        // -> repeat (B, L2, L1, H)
        auto dec_trans = mx::repeat(prefix+"_decoder_transformed",
                mx::expand_dims(
                mx::transpose(decoder_output, mx::Shape(1, 0 ,2)),2),
                encoder_max_sequence_length, dmlc::optional<int>(2));

        std::vector<mx::Symbol> to_concat;
        to_concat.push_back(dec_trans), to_concat.push_back(enc_trans);
        auto dec_enc_cat = mx::Concat(to_concat, to_concat.size(), 3);

        // declare variables for W and v in the formula

        if(params.find(prefix+"_W") == params.end())
            params[prefix+"_W"] = mx::Symbol::Variable(prefix+"_W");
        if(params.find(prefix+"_v") == params.end())
            params[prefix+"_v"] = mx::Symbol::Variable(prefix+"_v");

        //std::vector<mx::Symbol> to_squeeze;

        auto to_squeeze = (mx::FullyConnected(
                mx::tanh(mx::FullyConnected(dec_enc_cat, params[prefix+"_W"],
                                            mx::Symbol(), hidden_dimension, true, false)),
                params[prefix+"_v"], mx::Symbol(), 1, true, false));
        scores = mx::squeeze(prefix+"_squeezed", to_squeeze)[0];

    } else if (mode == AttentionMode::kGeneral) {

        // declare variable for the new metric matrix W
        if(params.find(prefix+"_W") == params.end())
            params[prefix+"_W"] = mx::Symbol::Variable(prefix+"_W");

        // encoder_output: (L1, B, H) -> (B, H, L1)
        // decoder_output: (L2, B, H) -> (B, L2, H)
        // then batch dot product to get (B, L2, L1)
        auto enc_trans = mx::transpose(prefix+"_encoder_transposed",
                                       encoder_output, mx::Shape(1, 2, 0));
        auto dec_trans = mx::transpose(prefix+"_decoder_transposed",
                                       decoder_output, mx::Shape(1, 0, 2));

        // we now compute the dot product (batch-wide) <x, Wy>; it does not
        // really matter if in the paper one applies to encoder or decoder

        scores = mx::batch_dot(prefix+"_general_dot_product",
                mx::FullyConnected(dec_trans, params[prefix+"_W"],
                mx::Symbol(), hidden_dimension, true, false),
                enc_trans);
    } else {
        throw dmlc::Error("Invalid choice of mode for attention");
    }


    // apply the softmax along the dimension L1 to get attention weights
    // for the encoder output

    auto scores_softmax = mx::softmax(prefix+"_scores_softmax", scores, mx::Symbol());
    // the scores are now used to give weight to the components of the encoder
    // output
    // encoder_output: (L1, B, H) -> (B, L1, H)
    // scores: (B, L2, L1)
    // batch dot product to land on (B, L2, H)
    auto enc_trans_1 = mx::transpose(prefix+"encoder_transposed_1",
                                     encoder_output, mx::Shape(1, 0, 2));

    std::vector<mx::Symbol> concat_input_vector;
    concat_input_vector.push_back(
            mx::batch_dot(prefix+"_context", scores_softmax,
            enc_trans_1));

    // decoder_output: (L2, B, H)
    // context: (B, L2, H)
    // transpose and concat to get (B, L2, 2H)
    concat_input_vector.push_back(mx::transpose(prefix+"decoder_transposed_1",
                                     decoder_output, mx::Shape(1, 0, 2)));

    auto concat_input = mx::Concat(prefix+"_concatenated_input",
            concat_input_vector, 2, 2);

    // apply tanh
    auto concat_output = mx::tanh(prefix+"_concatenated_output", concat_input);

    // the concatenation weight is used to construct  the new
    // hidden state using matrix multiplication
    if(params.find(prefix+"_concatenation_weight") == params.end())
        params[prefix+"_concatenation_weight"] = mx::Symbol::Variable(prefix+"_concatenation_weight");

    // note the reshape as the concatenation weight will no longer look at the interactions in the L2-dimension
    // on L1 before was ok because we can see all the input sentence
    auto transformed_hidden = mx::Reshape(mx::FullyConnected(mx::Reshape(concat_output, mx::Shape(B*L2, 2*H)),
            params[prefix+"_concatenation_weight"],
            mx::Symbol(), hidden_dimension, true, false), mx::Shape(B, L2, H));

    return std::make_pair(params, transformed_hidden);
}

