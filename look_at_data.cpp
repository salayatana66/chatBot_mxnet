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
#include<getopt.h>
#include "checkpointing.h"
using namespace mxnet::cpp;


//todo

int main(int argc, char **argv) {

    /** parsing arguments
     *  initialization with default values
     */

    uint32_t batch_size = 64;
    // internal state size is the same as the size of the
    // word embeddings
    uint32_t state_size = 500;
    // nr of layers in encoder & decoder
    uint32_t num_layers = 2;
    // dropout
    mx_float dropout = 0.1;
    // attention mode
    AttentionMode attentionMode = AttentionMode::kDot;

    // parameters for logging
    int epochs = 2;
    int print_each_n_iterations = 100;

    // the decoder optimizer uses a different lr obtained multiplying the
    // encoder by decoder_learning_ratio
    double encoder_learning_rate = 1e-4, clip_parameter = 50.0;
    double decoder_learning_ratio = 5.0;

    // logging file
    auto logging_file_name = std::string();
    std::ofstream outputFile;

    // checkpointing string
    auto checkpointing_file_name = std::string();

    static struct option long_options[] = {
     {"batch-size",   required_argument, nullptr,  0 },
     {"state-size",   required_argument, nullptr,  0 },
     {"num-layers", required_argument, nullptr, 0},
     {"dropout", required_argument, nullptr, 0},
     {"attention-mode", required_argument, nullptr, 0},
     {"epochs", required_argument, nullptr, 0},
     {"print-each-n-iterations", required_argument, nullptr, 0},
     {"encoder-learning-rate", required_argument, nullptr, 0},
     {"clip-parameter", required_argument, nullptr, 0},
     {"decoder-learning-ratio", required_argument, nullptr, 0},
     {"logging-file", required_argument, nullptr, 0},
     {"checkpoint-file", required_argument, nullptr, 0}
    };

    int c;
    while (true) {
        int option_index = 0;
        const char *option_name;
        c = getopt_long(argc, argv, "",
                        long_options, &option_index);
        if (c == -1)
            break;
        else if (c == 0) {
            option_name = long_options[option_index].name;
            if (std::string(option_name) == "batch-size") {
                batch_size = atoi(optarg);
                printf("Setting batch size to %d\n", batch_size);
            } else if (std::string(option_name) == "state-size") {
                state_size = atoi(optarg);
                printf("Setting latent embedding dimension to %d\n", state_size);
            } else if (std::string(option_name) == "num-layers") {
                num_layers = atoi(optarg);
                printf("Setting number of layers in encoder & decoder to %d\n", num_layers);
            } else if (std::string(option_name) == "dropout") {
                dropout = atof(optarg);
                printf("Setting dropout to %f\n", dropout);
            } else if (std::string(option_name) == "attention-mode") {
                if (std::string(optarg) == "dot") {
                    attentionMode = AttentionMode::kDot;
                    printf("Setting attention mode to %s\n", optarg);
                } else if (std::string(optarg) == "concat") {
                    attentionMode = AttentionMode::kConcat;
                    printf("Setting attention mode to %s\n", optarg);
                } else if (std::string(optarg) == "general") {
                    attentionMode = AttentionMode::kGeneral;
                    printf("Setting attention mode to %s\n", optarg);
                } else {
                    printf("Invalid attention mode %s was specified\n", optarg);
                    return -1;
                }
            } else if (std::string(option_name) == "epochs") {
                epochs = atoi(optarg);
                printf("Training for # of epochs %d\n", epochs);
            } else if (std::string(option_name) == "print-each-n-iterations") {
                print_each_n_iterations = atoi(optarg);
                printf("Printing loss each n-iterations %d\n", print_each_n_iterations);
            } else if (std::string(option_name) == "encoder-learning-rate") {
                encoder_learning_rate = atof(optarg);
                printf("Setting encoder learning rate to %f\n", encoder_learning_rate);
            } else if (std::string(option_name) == "clip-parameter") {
                clip_parameter = atof(optarg);
                printf("Setting gradient clip parameter %f\n", clip_parameter);
            } else if (std::string(option_name) == "decoder-learning-ratio") {
                decoder_learning_ratio = atof(optarg);
                printf("Setting decoder / encoder learning ratio to %f\n", decoder_learning_ratio);
            } else if (std::string(option_name) == "logging-file") {
                logging_file_name = std::string(optarg);
                printf("Using logging file %s\n", logging_file_name.c_str());
                outputFile.open(logging_file_name);
            } else if (std::string(option_name) == "checkpoint-file") {
                checkpointing_file_name = std::string(optarg);
                printf("Using checkpoint file %s\n", checkpointing_file_name.c_str());
            } else {
                option_name = long_options[option_index].name;
                printf("Unknown option %s", option_name);
                printf("\n");
                return -1;
            }


        }
    }

    std::cout << "Using batch size " << batch_size <<
    std::endl << "Using embedding dimension " << state_size <<
    std::endl << "Using number of layers " << num_layers <<
    std::endl << "Using dropout " << dropout <<
    std::endl << "Using attention mode " << attentionMode <<
    std::endl << "Training for " << epochs << " epochs" <<
    std::endl << "Printing loss function every " << print_each_n_iterations <<
    " iterations" << std::endl <<
    "encoder_learning_rate: " << encoder_learning_rate << ", decoder learning RATIO: "
    << decoder_learning_ratio << ", gradient clipping parameter: " << clip_parameter <<
    std::endl;

    if(!checkpointing_file_name.empty())
        std::cout << "After each epoch saving checkpoint to " <<
        checkpointing_file_name << std::endl;

    if(outputFile.is_open()) {
        std::cout << "Using " << logging_file_name << " for logging"
        << std::endl;

        outputFile << "Using batch size " << batch_size <<
                  std::endl << "Using embedding dimension " << state_size <<
                  std::endl << "Using number of layers " << num_layers <<
                  std::endl << "Using dropout " << dropout <<
                  std::endl << "Using attention mode " << attentionMode <<
                  std::endl << "Training for " << epochs << " epochs" <<
                  std::endl << "Printing loss function every " << print_each_n_iterations <<
                  " iterations" << std::endl <<
                  "encoder_learning_rate: " << encoder_learning_rate << ", decoder learning RATIO: "
                  << decoder_learning_ratio << ", gradient clipping parameter: " << clip_parameter <<
                  std::endl;

        if(!checkpointing_file_name.empty())
            outputFile << "After each epoch saving checkpoint to " <<
                      checkpointing_file_name << std::endl;
    }

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end - start;

    // get ref to CPU
    Context device(DeviceType::kGPU, 0);
    Context cpuDevice(DeviceType::kCPU, 0);

    auto batcher = BatchIter(device, batch_size, "data/vocabulary.txt", "data/cleaned_conversations.txt");

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;

    std::cout << "Finished Loading sencence pairs " <<
              elapsed_seconds.count() << 's' << std::endl;


    // create vectors containing the trainable parameters
    // classifying them in encoder & decoder
    // note also that some parameters like the word embeddings might
    // be shared
    std::vector<std::string> enc_trainable_parameters;
    std::vector<std::string> dec_trainable_parameters;

    // word indices for the input & output sentences & for the decoder feed
    // the lagged one
    auto input_sentence_data = Symbol::Variable("input_sentence_data");
    auto output_sentence_data = Symbol::Variable("output_sentence_data");
    auto lagged_output_sentence_data = Symbol::Variable("lagged_output_sentence_data");

    // weights for the word embeddings, shared across encoder & decoder
   auto vocabulary_embedding_weight = Symbol::Variable("vocabulary_embedding_weight");
    enc_trainable_parameters.emplace_back("vocabulary_embedding_weight");
    dec_trainable_parameters.emplace_back("vocabulary_embedding_weight");


    // mapping sentences to embeddings for feeding encoder & decoder
    auto input_sentence_embedding = Embedding("input_sentence_embedding",
                                          transpose(input_sentence_data),
                                          vocabulary_embedding_weight, batcher.vocabulary_size(),
                                          state_size);
    auto output_sentence_embedding = Embedding("output_sentence_embedding",
            transpose(lagged_output_sentence_data),
            vocabulary_embedding_weight, batcher.vocabulary_size(),
            state_size);


    // definining the decoder symbol
    auto encoder_params = mx::Symbol::Variable("encoder_params");
    enc_trainable_parameters.emplace_back("encoder_params");

    auto encoder_sequence_length = mx::Symbol::Variable("encoder_sequence_length");
    auto encoder_initial_state = mx::Symbol::Variable("encoder_initial_state");
    auto encoder = createEncoder("encoder", state_size, num_layers,
            input_sentence_embedding, encoder_sequence_length, encoder_initial_state,
            encoder_params, dropout);

    /** encoder outputs consist of
     * [0] -> outputs of shape [T, B, 2*hidden_size]
     * [1] -> hidden state of shape [2*layers, B, hidden_size]
     *
     * to recover the dimension to feed the bidirectional output state
     * into the mono-directional decoder we sum the states
     */

    auto encoder_output = slice_axis(encoder[0], 2, 0, dmlc::optional<int>(state_size)) +
            slice_axis(encoder[0], 2, state_size, dmlc::optional<int>());
    auto encoder_hidden_state = encoder[1];


    auto decoder_params = mx::Symbol::Variable("decoder_params");
    dec_trainable_parameters.emplace_back("decoder_params");

    auto decoder_sequence_length = mx::Symbol::Variable("decoder_sequence_length");
    auto decoder = createDecoder("decoder", state_size, 2*num_layers,
            output_sentence_embedding, encoder_hidden_state, decoder_sequence_length,
            decoder_params, dropout);
    auto decoder_output = decoder[0];


    // definining attention
   std::map<std::string, Symbol> attention_params;

    auto attention_pair = createAttention("attention", state_size, batcher.max_first_sentence_len(),
           batcher.max_second_sentence_len(), batch_size, encoder_output, decoder_output, attention_params, attentionMode);

    // attention is considered part of the decoder parameters
   for(const auto& param: attention_params)
       dec_trainable_parameters.emplace_back(param.first);

   auto attention_hidden_state = attention_pair.second;

   // compute the output values for each work in the vocabulary
   auto decoding_weights = Symbol::Variable("decoding_weights");
   dec_trainable_parameters.emplace_back("decoding_weights");


    //decoding layer for the embeddings
   auto decoding = mx::Reshape(mx::FullyConnected(mx::Reshape(attention_hidden_state, mx::Shape(batch_size*batcher.max_second_sentence_len(),
           state_size)), decoding_weights,
           mx::Symbol(), batcher.vocabulary_size(), true, false),
                   mx::Shape(batch_size, batcher.max_second_sentence_len(), batcher.vocabulary_size()));

   // we now apply the softmax to get the probabilities of the output words
   auto output_words_probability = mx::softmax("next_word_probability",
           decoding, mx::Symbol(), 2);

 /**
  * the word index corresponding to the probability is on axis 2
    however mxnet has some limitations with the current slicing
    so we start with output_words_probability -> (B, T, V),
    V being the vocabulary size
    output_sentence_data -> (B, T)
    we need to use pick, but first reshape
    output_words_probability -> (B*T, V)
    output_sentence_data -> (B*T)
    so we lookup to get rid of V and then reshape to (B, T)
    */
   auto output_prob_at_labels = mx::Reshape("output_prob_at_labels",
           mx::pick(
           mx::Reshape(output_words_probability,
           Shape(batch_size*batcher.max_second_sentence_len(), batcher.vocabulary_size())),
           mx::Reshape(output_sentence_data, Shape(batch_size*batcher.max_second_sentence_len()))),
                   Shape(batch_size, batcher.max_second_sentence_len()));

   // to compute the cross entropy we need to stabilize the logarithms

    auto log_stabilizer = mx::Symbol::Variable("log_stabilizer");

    // in reality this is the negative of the cross entropy
    // for each element
    auto cross_entropy = mx::log(output_prob_at_labels + log_stabilizer);

    // to store the mask
    auto output_mask = mx::Symbol::Variable("output_mask");

    auto masked_cross_entropy = mx::sum(output_mask*cross_entropy)/mx::sum(output_mask);

    auto final_cross_entropy = mx::negative(masked_cross_entropy);
    auto loss = mx::MakeLoss(final_cross_entropy);

    /**
     * This is used for Shape inference
     */
    std::map<std::string, NDArray> args_map;

    args_map["input_sentence_data"] = NDArray(Shape(batch_size, batcher.max_first_sentence_len()), device, false);
    args_map["output_sentence_data"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);
    args_map["lagged_output_sentence_data"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);

    args_map["encoder_inital_state"] = NDArray(Shape(2 * num_layers, batch_size, state_size), device, false);
    args_map["encoder_sequence_length"] = NDArray(Shape(batch_size), device, false);
    args_map["decoder_sequence_length"] = NDArray(Shape(batch_size), device, false);
    args_map["log_stabilizer"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);

    args_map["output_mask"] = NDArray(Shape(batch_size, batcher.max_second_sentence_len()), device, false);

    // We instance the executor
    Executor *exe = loss.SimpleBind(device, args_map);

    // allocate data to initialize the encoder state to 0
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

    // allocate data to set the log stabilizer to 1e-15
    mx_float *const_log_stabilizer = new mx_float[batch_size*batcher.max_second_sentence_len()];
    auto *p_log_stabilizer = const_log_stabilizer;
    for(int i=0; i<batch_size*batcher.max_second_sentence_len(); i++) {
        *p_log_stabilizer = 1e-15;
        p_log_stabilizer++;
    }
    auto nd_log_stabilizer = NDArray(Shape(batch_size, batcher.max_second_sentence_len()),
            device, false);
    nd_log_stabilizer.SyncCopyFromCPU(const_log_stabilizer, batch_size*batcher.max_second_sentence_len());

    // initialize vocabulary_embedding_weight

    auto vocabulary_embedding_initializer = mx::Uniform(1.0/std::sqrt(state_size));
    vocabulary_embedding_initializer("vocabulary_embedding_weight",
            &(exe->arg_dict()["vocabulary_embedding_weight"]));

    // initialize parameters for encoder, note use of Xavier initialization
    // inside the neural network
    for(const auto& param: enc_trainable_parameters) {
        if (param != "vocabulary_embedding_weight") {
            if ((param != "encoder_params") & (param != "decoder_params")) {
                auto uniformInitializer = mx::Uniform(1.0);

                std::cout << "Initializing " << param << " via U[-1.0, 1.0] " <<
                          std::endl;

                uniformInitializer(param, &(exe->arg_dict()[param]));
            } else if ((param == "encoder_params") | (param == "decoder_params")) {
                auto xavierInitializer = mx::Xavier(mx::Xavier::RandType::uniform,
                                                    mx::Xavier::FactorType::avg, 1.0);

                std::cout << "Initializing " << param << " via Xavier " << std::endl;

                xavierInitializer(param, &(exe->arg_dict()[param]));

            } else {
                std::cout << "Skipping initialization for " << param << std::endl;
            }
        }
    }

    // initialize parameters for decoder, note use of Xavier initialization
    // inside the neural network
    for(const auto& param: dec_trainable_parameters) {
        if (param != "vocabulary_embedding_weight") {
            if ((param != "encoder_params") & (param != "decoder_params")) {
                auto uniformInitializer = mx::Uniform(1.0);

                std::cout << "Initializing " << param << " via U[-1.0, 1.0] " <<
                          std::endl;

                uniformInitializer(param, &(exe->arg_dict()[param]));
            } else if ((param == "encoder_params") | (param == "decoder_params")) {
                auto xavierInitializer = mx::Xavier(mx::Xavier::RandType::uniform,
                                                    mx::Xavier::FactorType::avg, 1.0);

                std::cout << "Initializing " << param << " via Xavier " << std::endl;

                xavierInitializer(param, &(exe->arg_dict()[param]));

            } else {
                std::cout << "Skipping initialization for " << param << std::endl;
            }
        }
    }


    // create a vector on CPU to periodically visualize
    // outputs from CPU
    std::vector<NDArray> outputs_on_cpu;
    for(auto i=0; i<exe->outputs.size(); ++i) {
        outputs_on_cpu.emplace_back(NDArray(exe->outputs[i].GetShape(),
                cpuDevice, false));
    }


    // the encoder optimizer is just Adam
    Optimizer* encoder_opt = new AdamOptimizer();
    encoder_opt->SetParam("lr", encoder_learning_rate)
    ->SetParam("clip_gradient", clip_parameter);

    // the decoder optimizer uses a different lr obtained multiplying the
    // encoder by decoder_learning_ratio
    Optimizer* decoder_opt = new AdamOptimizer();
    decoder_opt->SetParam("lr", decoder_learning_ratio*encoder_learning_rate)
    ->SetParam("clip_gradient", clip_parameter);

    // create vectors to apply steps for decoder & encoder separately
    std::vector<int> dec_param_index, enc_param_index;
    int arg_index = 0;
    for(const auto& argument: loss.ListArguments()) {
        std::cout << "Obtained parameter #" << arg_index << " named " <<
        argument << std::endl;
        if(std::find(enc_trainable_parameters.begin(),
                enc_trainable_parameters.end(), argument) != enc_trainable_parameters.end()) {
            enc_param_index.emplace_back(arg_index);
            std::cout << "Assigning parameter named " << argument << " to encoder optimizer " << std::endl;
        }

        if(std::find(dec_trainable_parameters.begin(),
                     dec_trainable_parameters.end(), argument) != dec_trainable_parameters.end()) {
            enc_param_index.emplace_back(arg_index);
            std::cout << "Assigning parameter named " << argument << " to decoder optimizer " << std::endl;
        }

        arg_index++;
    }


    //todo
    // 1. try different Luong attentions
    // 2. rename look at data
    // 3. script to generate sentences via greedy search

    int current_epoch = 0;
    int iteration_counter = 0;
    mx_float rolling_loss = 0;

    while(current_epoch < epochs) {
        // todo: possibly remove reload from checkpoint
        if((current_epoch > 0) & !(checkpointing_file_name.empty())) {
            LoadCheckpoint(checkpointing_file_name+".encoder", exe);
            LoadCheckpoint(checkpointing_file_name+".decoder", exe);
        }

        start = std::chrono::steady_clock::now();
        std::cout << "Starting epoch " << (current_epoch+1) << std::endl;
        if(outputFile.is_open())
            outputFile << "Starting epoch " << (current_epoch+1) << std::endl;
        while (!batcher.am_exhausted()) {
            auto data_batch = batcher.getBatch();
            data_batch.input_sentence.CopyTo(&exe->arg_dict()["input_sentence_data"]);
            data_batch.input_sequence_lengths.CopyTo(&exe->arg_dict()["encoder_sequence_length"]);

            data_batch.output_sentence.CopyTo(&exe->arg_dict()["output_sentence_data"]);
            data_batch.lagged_output_sentence.CopyTo(&exe->arg_dict()["lagged_output_sentence_data"]);
            data_batch.output_sequence_lengths.CopyTo(&exe->arg_dict()["decoder_sequence_length"]);

            data_batch.output_mask.CopyTo(&exe->arg_dict()["output_mask"]);

            // remember to initialize the init state to 0
            nd_init_state.CopyTo(&exe->arg_dict()["encoder_initial_state"]);
            nd_log_stabilizer.CopyTo(&exe->arg_dict()["log_stabilizer"]);

            exe->Forward(true);
            exe->Backward();

            // update parameters
            for(auto i: enc_param_index) {
                encoder_opt->Update(i, exe->arg_arrays[i], exe->grad_arrays[i]);
            }
            for(auto i: dec_param_index) {
                decoder_opt->Update(i, exe->arg_arrays[i], exe->grad_arrays[i]);
            }

            // wait to synch results
            NDArray::WaitAll();

            if ((iteration_counter + 1) % print_each_n_iterations == 0) {
                auto times_logged = (iteration_counter + 1) / print_each_n_iterations;
                exe->outputs[0].CopyTo(&outputs_on_cpu[0]);
                NDArray::WaitAll();

                if(times_logged == 1) rolling_loss = *(outputs_on_cpu[0].GetData());
                else rolling_loss += (*(outputs_on_cpu[0].GetData()) -
                        rolling_loss) / times_logged;
                std::cout << "loss[" << (iteration_counter+1) <<
                " @Epoch " << (current_epoch+1) << "]= " <<
                          rolling_loss << std::endl;
                if(outputFile.is_open()) {
                    outputFile << "loss[" << (iteration_counter+1) <<
                                  " @Epoch " << (current_epoch+1) << "]= " <<
                                  rolling_loss << std::endl;
                }
            }

            iteration_counter++;

        }
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;

        std::cout << "Finished epoch " << (current_epoch+1)
        << " in " << elapsed_seconds.count() << "s" <<
        std::endl;

        if(outputFile.is_open()) {
            outputFile << "Finished epoch " << (current_epoch+1)
                       << " in " << elapsed_seconds.count() << "s" <<
                       std::endl;
        }
        current_epoch++;

        batcher.reset();

        if(!checkpointing_file_name.empty()) {
            SaveCheckpoint(checkpointing_file_name + ".encoder",
                    enc_trainable_parameters, exe);
            SaveCheckpoint(checkpointing_file_name + ".decoder",
                    dec_trainable_parameters, exe);
        }


    }

    if(outputFile.is_open()) outputFile.close();

    delete exe;
    delete decoder_opt;
    delete encoder_opt;
    return 0;
}

