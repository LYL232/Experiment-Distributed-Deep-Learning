//
// Created by LYL232 on 2021/2/11.
//

#include "mpi.h"
#include <stdexcept>
#include <iostream>
#include "global/Global.h"
#include "global/initialize.h"
#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/backend/mpi/MPICommunicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::mutex MPIBackend::mutex_;
bool MPIBackend::initialized_ = false;
bool MPIBackend::finalized_ = false;
int MPIBackend::refs_ = 0;
// todo: 这里不能设成常量
//const size_t MPIBackend::maxMessageTransferSize = ((size_t)(1) << 31) - 1;


MPIBackend::MPIBackend(int *argc, char ***argv) : CommunicationBackend() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (finalized_) {
        std::cerr << "ERROR: trying constructing MPIBackend after MPI_Finalize" << std::endl;
        throw std::runtime_error(
                "trying constructing MPIBackend after MPI_Finalize"
        );
    }
    if (!initialized_) {
        initialize_(argc, argv);
    }
    refs_++;
}

MPIBackend::~MPIBackend() {
    std::lock_guard<std::mutex> guard(mutex_);
    --refs_;
    if (refs_ == 0) {
        if (!finalized_) {
            MPI_Finalize();
            finalized_ = true;
        }
    }
}

MPI_Datatype MPIBackend::DataType2MPIType(DataType dtype) noexcept {
    using namespace tensorflow;
    switch (dtype) {
        case DT_FLOAT:
            return MPI_FLOAT;
        case DT_DOUBLE:
            return MPI_DOUBLE;
        case DT_INT32:
            return MPI_INT;
        case DT_INT64:
            return MPI_INT64_T;
        case DT_UINT64:
            return MPI_UINT64_T;
        default:
            break;
    }
    GLOBAL_ERROR_WITH_THREAD_ID("trying getting unsupported DataType: " << dtype)
    return MPI_DATATYPE_NULL;
}


std::shared_ptr<Communicator> MPIBackend::worldGetter_(int *argc, char ***argv) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        initialize_(argc, argv);
    }
    return world_;
}

void MPIBackend::initialize_(int *argc, char ***argv) {
    int provided, required = MPI_THREAD_MULTIPLE;
    std::cerr << "MPI initializing" << std::endl;
    MPI_Init_thread(argc, argv, required, &provided);
    if (provided < required) {
        std::cerr << "ERROR: environment dose not provide mpi thread requirement" << std::endl;
        throw std::runtime_error(
                "environment dose not provide MPI_THREAD_MULTIPLE requirement"
        );
    }
    auto *copiedWorld = new MPI_Comm(MPI_COMM_WORLD);  // no mem track
    int rank, size;
    MPI_Comm_rank(*copiedWorld, &rank);
    MPI_Comm_size(*copiedWorld, &size);
    std::cerr << "MPI initialized world rank: " << rank << ", size: " << size << std::endl;
    world_.reset(new MPICommunicator(  // no mem track
            std::shared_ptr<MPI_Comm>(copiedWorld),
            rank, size
    ));
    initialized_ = true;
}

std::shared_ptr<Communicator> MPIBackend::worldCommunicator() const noexcept {
    return worldGetter_(nullptr, nullptr);
}


}}}


