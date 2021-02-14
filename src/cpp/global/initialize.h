//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H

#include <fstream>
#include "global/GlobalLog.h"
#include "communicate/collective/controller/TensorsCollectiveCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::shared_ptr<GlobalLog> globalLogGetter();

std::shared_ptr<CommunicationBackend> communicationBackendGetter();

std::shared_ptr<TensorsCollectiveCommunicateController> collectiveCommunicateControllerGetter();

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H
