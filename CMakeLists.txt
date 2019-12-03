cmake_minimum_required(VERSION 3.13)
project(chat_bot_provisory)

set(CMAKE_CXX_STANDARD 14)
#add_compile_options(
#        "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
#        "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>"
#)
include_directories("/mnt/reclaimed/remote-aws-mxnet/cpp-package/include/"
        "/mnt/reclaimed/remote-aws-mxnet/include/")
#todo need to refactor the library into less components
add_library(data_processing data_processing.cpp)
add_library(data_loading data_loading.cpp)
add_library(data_representation data_representation.cpp)
add_library(models models.cpp)
add_library(checkpointing checkpointing.cpp)
add_executable(look_at_data look_at_data.cpp)
add_executable(rnn_arch rnn_arch.cpp)
add_executable(format_conversations format_conversations.cpp)
add_executable(create_vocabulary create_vocabulary.cpp)
add_executable(example_save_to_bytes example_save_to_bytes.cpp)
add_executable(greedy_decoder greedy_decoder.cpp)
target_link_libraries(rnn_arch data_processing data_representation /mnt/reclaimed/remote-aws-mxnet/libmxnet.so)
#target_link_libraries(rnn_arch data_processing data_representation /usr/local/lib/python3.6/dist-packages/mxnet/libmxnet.so)
target_link_libraries(look_at_data checkpointing models data_processing data_loading data_representation  /mnt/reclaimed/remote-aws-mxnet/libmxnet.so)
#target_link_libraries(look_at_data models data_processing data_loading data_representation /mnt/reclaimed/mx-net-python-env/lib/python3.6/site-packages/mxnet/libmxnet.so)
target_link_libraries(example_save_to_bytes data_processing data_representation)
target_link_libraries(format_conversations data_processing)
target_link_libraries(create_vocabulary data_processing)
target_link_libraries(data_loading data_processing)
add_executable(gpu_simple_test gpu_simple_test.cpp)
target_link_libraries(gpu_simple_test models data_loading data_representation  /mnt/reclaimed/remote-aws-mxnet/libmxnet.so)
target_link_libraries(greedy_decoder checkpointing models data_processing data_loading data_representation  /mnt/reclaimed/remote-aws-mxnet/libmxnet.so)
