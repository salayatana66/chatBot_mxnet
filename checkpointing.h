//
// Created by arischioppa on 11/29/19.
//

#ifndef CHAT_BOT_PROVISORY_CHECKPOINTING_H
#define CHAT_BOT_PROVISORY_CHECKPOINTING_H

#include "mxnet-cpp/MxNetCpp.h"
#include<string>
namespace mx = mxnet::cpp;

void SaveCheckpoint(const std::string& filepath,
        const std::vector<std::string>& params, mx::Executor* exe);
void LoadCheckpoint(const std::string& filepath,
         mx::Executor* exe);



#endif //CHAT_BOT_PROVISORY_CHECKPOINTING_H
