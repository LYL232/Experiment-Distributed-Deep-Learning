//
// Created by LYL232 on 2021/3/23.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATECONTROLLER_H

#include "communicate/tensor/collective/controller/rtc/RingTokenCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class MPIRingTokenCommunicateController : public RingTokenCommunicateController {
public:
    MPIRingTokenCommunicateController() = default;

    MPIRingTokenCommunicateController(MPIRingTokenCommunicateController &&) = delete;

    MPIRingTokenCommunicateController(const MPIRingTokenCommunicateController &) = delete;

private:
    std::shared_ptr<RingTokenCommunicateHandler> newHandler(
            const std::shared_ptr<Communicator> &communicator) override;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATECONTROLLER_H
