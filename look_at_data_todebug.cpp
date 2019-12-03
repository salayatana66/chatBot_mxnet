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


//todo
// 1. feed lagged sentence as input to decoder -> DONE
// 2. add softmax with for output character -> WIP need to apply the mask & take a
// stable log; main issue with the lookup has been resolved... why do we need
// to lookup? because otherwise the mask cannot applied to a normal
// softmax in the mxnet default library
// 3. add the mask to compute loss
// 4. try to perform a few iterations to see the loss
// 5. need to add initialization for all the parameters
// 6. need to figure out why the initialization for some parameters fails
// 7. need to understand the parameters issue along the full network
// 8. need to understand one layer at a time what is going on here!!!

// todo refactor -> size of all embeddings
int main() {

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end - start;
    start = std::chrono::steady_clock::now();
    int iCount = 0;

    // get ref to CPU
    Context device(DeviceType::kCPU, 0);
    Context gpuDevice(DeviceType::kGPU, 0);

    uint32_t batch_size = 3;
    auto batcher = BatchIter(device, batch_size, "data/vocabulary.txt", "data/cleaned_conversations.txt");

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;

    std::cout << "Finished Loading sencence pairs " <<
              elapsed_seconds.count() << 's' << std::endl;


    // internal state size is the same as the size of the
    // word embeddings
    uint32_t state_size = 5;

    // create a vector containing the name of trainable parameters
    std::vector<std::string> trainable_parameters;

    // word indices for the input & output sentences & for the decoder feed
    // the lagged one
    auto input_sentence_data = Symbol::Variable("input_sentence_data");
    auto output_sentence_data = Symbol::Variable("output_sentence_data");
    auto lagged_output_sentence_data = Symbol::Variable("lagged_output_sentence_data");

    // weights for the word embeddings, shared across encoder & decoder
    auto vocabulary_embedding_weight = Symbol::Variable("vocabulary_embedding_weight");
    trainable_parameters.push_back("vocabulary_embedding_weight");

    // mapping sentences to embeddings for feeding encoder & decoder
    auto input_sentence_embedding = Embedding("input_sentence_embedding",
                                          transpose(input_sentence_data),
                                          vocabulary_embedding_weight, batcher.vocabulary_size(),
                                          state_size);
    auto output_sentence_embedding = Embedding("output_sentence_embedding",
            transpose(lagged_output_sentence_data),
            vocabulary_embedding_weight, batcher.vocabulary_size(),
            state_size);

    // nr of layers in encoder & decoder
    uint32_t num_layers = 2;

    auto encoder_params = mx::Symbol::Variable("encoder_params");
    trainable_parameters.push_back("encoder_params");

    auto encoder_sequence_length = mx::Symbol::Variable("encoder_sequence_length");
    auto encoder_initial_state = mx::Symbol::Variable("encoder_initial_state");
    auto encoder = createEncoder("encoder", state_size, num_layers,
            input_sentence_embedding, encoder_sequence_length, encoder_initial_state,
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


    auto decoder_params = mx::Symbol::Variable("decoder_params");
    trainable_parameters.push_back("decoder_params");

    auto decoder_sequence_length = mx::Symbol::Variable("decoder_sequence_length");
   auto decoder = createDecoder("decoder", state_size, 2*num_layers,
            output_sentence_embedding, encoder_hidden_state, decoder_sequence_length,
            decoder_params);
   auto decoder_output = decoder[0];


   // try to see if we can feed the encoder output into the decoder

   std::map<std::string, Symbol> attention_params;

    auto attention_pair = createAttention("attention", state_size, batcher.max_first_sentence_len(),
           batcher.max_second_sentence_len(), batch_size, encoder_output, decoder_output, attention_params, AttentionMode::kDot);

   for(auto param: attention_params) trainable_parameters.push_back(param.first);

   auto attention_hidden_state = attention_pair.second;

   // compute the output values for each work in the vocabulary
   auto decoding_weights = Symbol::Variable("decoding_weights");
   trainable_parameters.push_back("decoding_weights");


   // decoding layer for the embeddings
   auto decoding = mx::Reshape(mx::FullyConnected(mx::Reshape(attention_hidden_state, mx::Shape(batch_size*batcher.max_second_sentence_len(),
           state_size)), decoding_weights,
           mx::Symbol(), batcher.vocabulary_size(), true, false),
                   mx::Shape(batch_size, batcher.max_second_sentence_len(), batcher.vocabulary_size()));

   // we now apply the softmax to get the probabilities of the output words
   auto output_words_probability = mx::softmax("next_word_probability",
           decoding, mx::Symbol(), 2);

   // the word index corresponding to the probability is on axis 2
   // however mxnet has some limitations with the current slicing
   // so we start with output_words_probability -> (B, T, V),
   // V being the vocabulary size
   // output_sentence_data -> (B, T)
   // we need to use pick, but first reshape
   // output_words_probability -> (B*T, V)
   // output_sentence_data -> (B*T)
   // so we lookup to get rid of V and then reshape to (B, T)
   auto output_prob_at_labels = mx::Reshape("output_prob_at_labels",
           mx::pick(
           mx::Reshape(output_words_probability,
           Shape(batch_size*batcher.max_second_sentence_len(), batcher.vocabulary_size())),
           mx::Reshape(output_sentence_data, Shape(batch_size*batcher.max_second_sentence_len()))),
                   Shape(batch_size, batcher.max_second_sentence_len()));

   // to compute the cross entropy we need to stabilize the logarithms

    auto log_stabilizer = mx::Symbol::Variable("log_stabilizer");


    auto cross_entropy = mx::log(output_prob_at_labels + log_stabilizer);

   auto output = mx::Symbol::Group(std::vector<mx::Symbol>({cross_entropy, input_sentence_data,
                                                            input_sentence_embedding}));



    std::map<std::string, NDArray> args_map;
    args_map["input_sentence_data"] = NDArray(Shape(batch_size, batcher.max_first_sentence_len()), device, false);
    args_map["output_sentence_data"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);
    args_map["encoder_inital_state"] = NDArray(Shape(2 * num_layers, batch_size, state_size), device, false);
    args_map["encoder_sequence_length"] = NDArray(Shape(batch_size), device, false);
    args_map["lagged_output_sentence_data"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);
    args_map["decoder_sequence_length"] = NDArray(Shape(batch_size), device, false);
    args_map["log_stabilizer"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);

    Executor *exe = output.SimpleBind(device, args_map);

    for(auto q: exe->arg_dict()) {
        std::cout << q.first << std::endl;
    }
    std::cout << "Number of outputs for Executor: " <<
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

    // set the log stabilizer to 1e-15
    mx_float *const_log_stabilizer = new mx_float[batch_size*batcher.max_second_sentence_len()];
    auto *p_log_stabilizer = const_log_stabilizer;
    for(int i=0; i<batch_size*batcher.max_second_sentence_len(); i++) {
        *p_log_stabilizer = 0.0000000001f;
        p_log_stabilizer++;
    }
    auto nd_log_stabilizer = NDArray(Shape(batch_size, batcher.max_second_sentence_len()),
            device, false);
    nd_log_stabilizer.SyncCopyFromCPU(const_log_stabilizer, batch_size*batcher.max_second_sentence_len());

    int test = 0;
    start = std::chrono::steady_clock::now();


    for (auto param : exe->arg_dict()) std::cout << param.first << std::endl;

    for(auto param: trainable_parameters) {
        if((param != "encoder_params") & (param != "decoder_params")) {
            auto uniformInitializer = mx::Uniform(1.0);

            std::cout << "Initializing " << param << " via U[-1.0, 1.0] " <<
            std::endl;

            uniformInitializer(param, &(exe->arg_dict()[param]));
        } else if ((param == "encoder_params") | (param == "decoder_params")) {
            auto xavierInitializer = mx::Xavier(mx::Xavier::RandType::uniform,
                    mx::Xavier::FactorType::avg, 1.0);

            std::cout << "Initializing " << param << " via Xavier " << std::endl;

            xavierInitializer(param, &(exe->arg_dict()[param]));

        }

        else {
            std::cout << "Skipping initialization for " << param << std::endl;
        }
    }

    std::vector<NDArray> outputs_on_cpu;
    for(int i=0; i<exe->outputs.size(); ++i) {
        outputs_on_cpu.push_back(NDArray(exe->outputs[i].GetShape(),
                gpuDevice, false));
    }

    while(!batcher.am_exhausted()) {
        auto data_batch = batcher.getBatch();
        data_batch.input_sentence.CopyTo(&exe->arg_dict()["input_sentence_data"]);
        data_batch.input_sequence_lengths.CopyTo(&exe->arg_dict()["encoder_sequence_length"]);
        // todo here needs the lagging to generate the data
        data_batch.output_sentence.CopyTo(&exe->arg_dict()["output_sentence_data"]);
        data_batch.lagged_output_sentence.CopyTo(&exe->arg_dict()["lagged_output_sentence_data"]);
        data_batch.output_sequence_lengths.CopyTo(&exe->arg_dict()["decoder_sequence_length"]);
        // remember to initialize the init state to 0
        nd_init_state.CopyTo(&exe->arg_dict()["encoder_initial_state"]);
        nd_log_stabilizer.CopyTo(&exe->arg_dict()["log_stabilizer"]);
        mx::NDArray::WaitAll();

        exe->Forward(false);

        for(int i=0; i<exe->outputs.size(); ++i) {
            exe->outputs[i].CopyTo(&outputs_on_cpu[i]);
        }
        mx::NDArray::WaitAll();

        for(int i=0; i<outputs_on_cpu.size(); ++i) {
            std::cout << "Step" << test << std::endl <<
            "Nr" << i << ": " << printNDArray(outputs_on_cpu[i])
            <<std::endl;
        }

        test++;
        if(test > 3) break;
        }


    end = std::chrono::steady_clock::now();

    elapsed_seconds = end - start;
    std::cout << "Time to loop over all batches " <<
    elapsed_seconds.count() << 's' << std::endl;

    return 0;
}

