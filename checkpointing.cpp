//
// Created by arischioppa on 11/29/19.
//

#include "checkpointing.h"

void SaveCheckpoint(const std::string& filepath,
                    const std::vector<std::string>& params, mx::Executor* exe) {
    std::map<std::string, mx::NDArray> outArray;
    for(auto& param: params) {
        outArray.insert({param, exe->arg_dict()[param]});
    }

    mx::NDArray::Save(filepath, outArray);
}

void LoadCheckpoint(const std::string& filepath,
                    mx::Executor* exe) {

    std::map<std::string, mx::NDArray> params = mx::NDArray::LoadToMap(filepath);
    mx::NDArray target;

    for(auto& param: params) {
        target = exe->arg_dict()[param.first];
        param.second.CopyTo(&target);
    }
}

