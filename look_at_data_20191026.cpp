#include<fstream>
#include<sstream>
#include<string>
#include<iostream>
#include<vector>
#include<queue>
#include<regex>
#include<stdio.h>
#include<chrono>
#include<exception>
#include "data_processing.h"
#include "data_representation.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "data_loading.h"
#include "simple_utils.h"
#include "models.h"
using namespace mxnet::cpp;

// todo refactor -> size of all embeddings
int HIDDEN_SIZE = 500;
int main() {

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end - start;
    start = std::chrono::steady_clock::now();
    int iCount = 0;

    // get ref to CPU
    Context device(DeviceType::kCPU, 0);
    uint32_t batch_size = 64;
    auto batcher = BatchIter(device, batch_size, "data/vocabulary.txt", "data/cleaned_conversations.txt");

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;

    std::cout << "Finished Loading sencence pairs " <<
              elapsed_seconds.count() << 's' << std::endl;


    // internal state size is the same as the size of the
    // word embeddings
    uint32_t state_size = 500;

    auto input_sentence_data = Symbol::Variable("data");
    auto vocabulary_embedding_weight = Symbol::Variable("vocabulary_embedding_weight");
    auto vocabulary_embedding = Embedding("vocabulary_embedding",
                                          transpose(input_sentence_data),
                                          vocabulary_embedding_weight, batcher.vocabulary_size(),
                                          state_size);

    uint32_t num_layers = 2;

    auto encoder_params = mx::Symbol::Variable("encoder_params");
    auto encoder_sequence_length = mx::Symbol::Variable("encoder_sequence_length");
    auto encoder_initial_state = mx::Symbol::Variable("encoder_initial_state");
    auto encoder = createEncoder("encoder", state_size, num_layers,
            vocabulary_embedding, encoder_sequence_length, encoder_initial_state,
            encoder_params);

    /** encoder outputs consist of
     * [0] -> outputs of shape [T, B, 2*hidden_size]
     * [1] -> hidden state of shape [2*layers, B, hidden_size]
     *
     * [0] need to add the bidirectional vectors into the
     * firt input vector for the decoder
     * [1] need to slice up to the number of layers used by the
     * decoder
     */

    auto encoder_output = slice_axis(encoder[0], 2, 0, dmlc::optional<int>(state_size)) +
            slice_axis(encoder[0], 2, state_size, dmlc::optional<int>());
    auto encoder_hidden_state = encoder[1];

    // todo rename vocabulary embedding
    auto decoder_input_data = Symbol("decoder_input_data");
    auto decoder_input_embedding = Embedding("decoder_input_embedding",
            transpose(decoder_input_data),
            vocabulary_embedding_weight, batcher.vocabulary_size(),
            state_size);

    auto decoder_params = mx::Symbol::Variable("decoder_params");
    auto decoder_sequence_length = mx::Symbol::Variable("decoder_sequence_length");
   auto decoder = createDecoder("decoder", state_size, 2*num_layers,
            decoder_input_embedding, encoder_hidden_state, decoder_sequence_length,
            decoder_params);
   auto decoder_output = decoder[0];


   // try to see if we can feed the encoder output into the decoder

   std::map<std::string, Symbol> attention_params;
   auto attention_pair = createAttention("attention", state_size, batcher.max_first_sentence_len(),
           batcher.max_second_sentence_len(), encoder_output, decoder_output, attention_params, AttentionMode::kGeneral);

   auto attention_hidden_state = attention_pair.second;

   // compute the output values for each work in the vocabulary
   auto decoding_weights = Symbol::Variable("decoding_weights");
   // decoding layer for the embeddings
   auto decoding = mx::FullyConnected(attention_hidden_state, decoding_weights,
           mx::Symbol(), batcher.vocabulary_size(), true, false);

   // we now apply the softmax to get the probabilities of the output words
   auto output_words_probability = mx::softmax("next_word_probability",
           decoding);

    std::map<std::string, NDArray> args_map;
    args_map["data"] = NDArray(Shape(batch_size, batcher.max_first_sentence_len()), device, false);
    args_map["encoder_inital_state"] = NDArray(Shape(2 * num_layers, batch_size, state_size), device, false);
    args_map["encoder_sequence_length"] = NDArray(Shape(batch_size), device, false);
    args_map["decoder_input_data"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);
    args_map["decoder_sequence_length"] = NDArray(Shape(batch_size), device, false);


    Executor *exe = output_words_probability.SimpleBind(device, args_map);

    for(auto q: exe->arg_dict()) {
        std::cout << q.first << std::endl;
    }
    std::cout << "Number of outputs for Encoder: " <<
    exe->outputs.size() << std::endl;
    for(int i = 0; i<exe->outputs.size(); i++) {
        std::cout << "Shape of output " << i << ": " <<
        printShape(exe->outputs[i].GetShape()) << std::endl;

    }

    // initialize the encoder state to 0
    auto hidden_state_vec_size = 2*num_layers*batch_size*state_size;
    mx_float *init_state = new mx_float[hidden_state_vec_size],
    *p_init_state = init_state;
    for(int i=0; i<hidden_state_vec_size; i++) {
        *p_init_state=0;
        p_init_state++;
    }
    auto nd_init_state = NDArray(Shape(2*num_layers, batch_size, state_size),
            device, false);
    nd_init_state.SyncCopyFromCPU(init_state,hidden_state_vec_size);

    start = std::chrono::steady_clock::now();
    while(!batcher.am_exhausted()) {
        auto data_batch = batcher.getBatch();
        data_batch.input_sentence.CopyTo(&exe->arg_dict()["data"]);
        data_batch.input_sequence_lengths.CopyTo(&exe->arg_dict()["encoder_sequence_length"]);
        // todo here needs the lagging to generate the data
        data_batch.output_sentence.CopyTo(&exe->arg_dict()["decoder_input_data"]);
        data_batch.output_sequence_lengths.CopyTo(&exe->arg_dict()["decoder_sequence_length"]);
        // remember to initialize the init state to 0
        nd_init_state.CopyTo(&exe->arg_dict()["encoder_initial_state"]);
        exe->Forward(true);
        }


    end = std::chrono::steady_clock::now();

    elapsed_seconds = end - start;
    std::cout << "Time to loop over all batches " <<
    elapsed_seconds.count() << 's' << std::endl;

    return 0;
}

