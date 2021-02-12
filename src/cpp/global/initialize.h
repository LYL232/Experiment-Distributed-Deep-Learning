//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H

#include <fstream>
#include "global/GlobalLogStream.h"
#include "communicate/tensor/allreduce/TensorsAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::shared_ptr<GlobalLogStream> globalLogStreamGetter();

std::shared_ptr<CommunicationBackend> communicationBackendGetter();

std::shared_ptr<tensorsallreduce::TensorsAllreduceController> allreduceControllerGetter();

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H
