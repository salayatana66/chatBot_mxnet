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
#include<random>

using namespace mxnet::cpp;

std::pair<NDArray, NDArray> getBatch(std::uniform_real_distribution<float>& dis,
        std::mt19937& gen, int batch_size, Context ctx) {
    mx_float* data = new mx_float[batch_size],
    *labels = new mx_float[batch_size],
    *p_data = data, *p_labels = labels;

    for(int i=0; i<batch_size; ++i) {
        *p_data = dis(gen);
        *p_labels = 2.0*(*p_data);
        p_data++;
        p_labels++;
    }

    auto dataArray = NDArray(Shape(batch_size), ctx, false);
    auto labelsArray = NDArray(Shape(batch_size), ctx, false);
    dataArray.SyncCopyFromCPU(data, batch_size);
    labelsArray.SyncCopyFromCPU(labels, batch_size);
    delete[] data;
    delete[] labels;
    return std::make_pair(dataArray, labelsArray);

}

int main() {

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(-1.0, 1.0);


    int batch_size=24;

    // get ref to CPU
    Context gpuDevice(DeviceType::kGPU, 0);
    Context cpuDevice(DeviceType::kCPU, 0);

    auto input_data = Symbol::Variable("input_data"),
    coefficient = Symbol::Variable("coefficient"),
    label = Symbol::Variable("label");

    auto prediction = mx::broadcast_mul(coefficient, input_data);

    // the loss must be explicitly defined
    auto output = mx::MakeLoss(mx::mean((label-prediction)*(label-prediction)));

    std::map<std::string, NDArray> args_map;
    args_map["input_data"] = NDArray(Shape(batch_size), cpuDevice, false);
    args_map["label"] = NDArray(Shape(batch_size), cpuDevice, false);
    args_map["coefficient"] = NDArray(Shape(1), cpuDevice, false);

    auto arg_names = output.ListArguments();

    Executor *exe = output.SimpleBind(cpuDevice, args_map);


    for(int i = 0; i<exe->outputs.size(); i++) {
        std::cout << "Shape[Executor[ " << i << "]]: " <<
        printShape(exe->outputs[i].GetShape()) << std::endl;

    }

    auto uniformInitializer = mx::Uniform(1.0);

    uniformInitializer("coefficient", &exe->arg_dict()["coefficient"]);

    Optimizer* opt = OptimizerRegistry::Find("sgd");
    opt->SetParam("lr", 0.1);

    for(int i=0; i<10; i++) {
        auto batchData = getBatch(dis, gen, batch_size, cpuDevice);
        batchData.first.CopyTo(&exe->arg_dict()["input_data"]);
        batchData.second.CopyTo(&exe->arg_dict()["label"]);

        NDArray::WaitAll();
        exe->Forward(true);
        exe->Backward();

        for(int i=0; i<arg_names.size(); ++i){
            if(arg_names[i] != "label" & arg_names[i] != "input_data") {
                opt->Update(i, exe->arg_arrays[i], exe->grad_arrays[i]);
            }
        }

        NDArray::WaitAll();
        std::cout << "I[" << i << "]: " << printNDArray(exe->outputs[0])
        << std::endl;

    }

    return 0;
}

