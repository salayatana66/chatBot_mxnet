//
// Created by arischioppa on 10/7/19.
//

// Simple utils to interact with mxnet

#ifndef CHAT_BOT_PROVISORY_SIMPLE_UTILS_H
#define CHAT_BOT_PROVISORY_SIMPLE_UTILS_H
#include "mxnet-cpp/MxNetCpp.h"
#include<string>

namespace mx = mxnet::cpp;

mx_float printNorm(const mx::NDArray &array) {
    mx_float out = 0;
    const mx_float* data = array.GetData();

    for(int i=0; i<array.Size(); ++i) {
        out += (*data)*(*data);
        data++;
    }
    return out;
}

std::string printNDArray(const mx::NDArray &array) {
    std::stringstream out;
    auto the_shape = array.GetShape();
    std::vector<int> dims;

    if(array.Size() == 1) {
        out << (*array.GetData());
        return out.str();
    }
    for(int i = 0; i < the_shape.size(); ++i) {
        dims.push_back(the_shape[i]);
    }

    // reverse size in products to decide end parentheses
    // moduli are used to decide when to create a parenthesis break
    std::vector<int> moduli;

    int prod = 1;
    for(auto i = dims.rbegin(); i != dims.rend(); ++i) {
        prod *= *i;
        moduli.push_back(prod);
    }
    const mx_float *p_data = array.GetData();

    for(int i = 0, j = array.Size(); i < j; ++i){
        for(auto q: moduli) {
            if((i + 1) % q == 1)
                out << '[';
        }
        out << *p_data;
        for(auto q: moduli) {
            if ((i + 1) % q == 0) {
                out << ']';
            }
        }
        p_data++;
        if(i < j - 1) out << ',';
    }
    return  out.str();

}


std::string printShape(mx::Shape s) {
    std::stringstream out;
    out << "(";
    for(int i=0; i<s.Size(); i++) {
        out << s[i];
        if(i < s.Size()-1) out << ",";
    }
    out << ")";
    return out.str();
}

std::string printShape(std::vector<mx_uint> v) {
    std::stringstream out;
    out << "(";
    for(int i=0; i<v.size(); i++) {
        out << v[i];
        if(i < v.size()-1) out << ",";
    }
    out << ")";
    return out.str();
}
#endif //CHAT_BOT_PROVISORY_SIMPLE_UTILS_H
