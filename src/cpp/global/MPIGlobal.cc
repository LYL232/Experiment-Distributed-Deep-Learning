//
// Created by LYL232 on 2021/2/10.
//

#include <fstream>
#include "global/MPIGlobal.h"
#include "communicate/communication/mpi/MPIBackend.h"
#include "communicate/tensor/allreduce/rta/MPIRingTokenAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl {

MPIGlobal MPIGlobal::instance_;

MPIGlobal::MPIGlobal() {
    init();
    assert(Global::get().initialized());
    assert(static_cast<MPIGlobal *>(&Global::get()) == &instance_);
}

void MPIGlobal::init() {
    auto backend = std::make_shared<MPIBackend>();
    int rank = backend->processRank();
    std::string logFile("log-");
    logFile.append(std::to_string(rank)).append(".txt");
    auto *log = new std::ofstream(logFile);
    std::function<void()> logStreamDestructor = [log]() {
        log->close();
        delete log;
    };
    initialize(
            this, log, logStreamDestructor, backend,
            new tensorsallreduce::rta::MPIRingTokenAllreduceController(
                    *log, backend)
    );
}

}}}