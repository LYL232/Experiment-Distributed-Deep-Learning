//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H

#include <fstream>
#include "global/GlobalLog.h"
#include "communicate/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/end2end/controller/TensorEnd2EndCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::shared_ptr<GlobalLog> globalLogGetter();

std::shared_ptr<CommunicationBackend> communicationBackendGetter();

std::shared_ptr<TensorsCollectiveCommunicateController> collectiveCommunicateControllerGetter();

std::shared_ptr<TensorEnd2EndCommunicateController> end2EndCommunicateControllerGetter();

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_INITIALIZE_H
