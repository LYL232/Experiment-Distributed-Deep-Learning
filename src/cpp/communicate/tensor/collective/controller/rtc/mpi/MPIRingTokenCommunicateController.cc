//
// Created by LYL232 on 2021/3/23.
//

#include "communicate/tensor/collective/controller/rtc/mpi/MPIRingTokenCommunicateController.h"
#include "communicate/tensor/collective/controller/rtc/mpi/MPIRingTokenCommunication.h"
#include "global/Global.h"


namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

std::shared_ptr<RingTokenCommunicateHandler>
MPIRingTokenCommunicateController::newHandler(const std::shared_ptr<Communicator> &communicator) {
    GLOBAL_INFO_WITH_THREAD_ID(
            "new RingTokenCommunicateHandler, communicator id: "
                    << communicator->id() << ", rank: " << communicator->rank())
    return std::make_shared<RingTokenCommunicateHandler>(
            communicator,
            std::make_shared<MPIRingTokenCommunication>(communicator)
    );
}

}}}}