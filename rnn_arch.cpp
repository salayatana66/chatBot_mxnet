#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

bool TIME_MAJOR = true;
Symbol LSTMWithBuiltInRNNOp(int num_lstm_layer, int sequence_length, int input_dim,
                            int num_hidden, int num_embed, mx_float dropout = 0) {
    auto isTrain = sequence_length > 1;
    auto data = Symbol::Variable("data");
    if (TIME_MAJOR && isTrain)
        data = transpose(data);

    auto embed_weight = Symbol::Variable("embed_weight");
    auto embed = Embedding("embed", data, embed_weight, input_dim, num_embed);
     if (!TIME_MAJOR && isTrain)
        embed = SwapAxis(embed, 0, 1);  // Change to time-major as cuDNN requires

    // We need not do the SwapAxis op as python version does. Direct and better performance in C++!
    auto rnn_h_init = Symbol::Variable("LSTM_init_h");
    auto rnn_c_init = Symbol::Variable("LSTM_init_c");
    auto rnn_params = Symbol::Variable("LSTM_parameters");  // See explanations near RNNXavier class
    auto variable_sequence_length = Symbol::Variable("sequence_length");
    auto rnn = RNN(embed, rnn_params, rnn_h_init, rnn_c_init, variable_sequence_length, num_hidden,
                   num_lstm_layer, RNNMode::kLstm, false, dropout, !isTrain);

    return  rnn;
}

int main() {

    Context device(DeviceType::kCPU, 0);

    int batch_size = 128,
            num_hidden = 200,
            num_embed = 200,
            num_lstm_layer = 2,
            input_dim = 600,
            vocab_len = 1500, sequence_length_max = 0;


    auto RNN = LSTMWithBuiltInRNNOp(num_lstm_layer, sequence_length_max, input_dim, num_hidden,
                                    num_embed, 0);
    std::map<std::string, NDArray> args_map;
    args_map["data"] = NDArray(Shape(batch_size, sequence_length_max), device, false);
    // Avoiding SwapAxis, batch_size is of second dimension.
    args_map["LSTM_init_c"] = NDArray(Shape(num_lstm_layer, batch_size, num_hidden), device, false);
    args_map["LSTM_init_h"] = NDArray(Shape(num_lstm_layer, batch_size, num_hidden), device, false);
    std::vector<mx_float> zeros(batch_size * num_lstm_layer * num_hidden, 0);
    Executor* exe = RNN.SimpleBind(device, args_map);

}