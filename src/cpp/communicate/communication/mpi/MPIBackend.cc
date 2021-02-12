//
// Created by LYL232 on 2021/2/11.
//

#include "mpi.h"
#include <stdexcept>
#include <iostream>
#include "global/Global.h"
#include "MPIBackend.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::mutex MPIBackend::mutex_;
bool MPIBackend::initialized_ = false;
bool MPIBackend::finalized_ = false;
int MPIBackend::processes_ = -1;
int MPIBackend::processRank_ = -1;
int MPIBackend::refs_ = 0;


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

StatusCode MPIBackend::allreduce(
        void *sendBuffer, void *recvBuffer,
        size_t elements, DataType dtype,
        AllreduceOperation op) const {
    MPI_Allreduce(
            sendBuffer, recvBuffer,
            (int) elements,
            DataType2MPIType(dtype),
            AllreduceOperation2MPIOp(op),
            MPI_COMM_WORLD
    );
    // todo: status check
    return STATUS_OK;
}

int MPIBackend::DataType2MPIType(DataType dtype) noexcept {
    using namespace tensorflow;
    switch (dtype) {
        case DT_FLOAT:
            return MPI_FLOAT;
        case DT_DOUBLE:
            return MPI_DOUBLE;
        case DT_INT32:
            return MPI_INT;
        case DT_INT64:
            return MPI_LONG_INT;
        default:
            break;
    }
    GLOBAL_ERROR_WITH_RANK_THREAD_ID("trying getting unsupported DataType: " << dtype);
    return MPI_DATATYPE_NULL;
}

MPI_Op MPIBackend::AllreduceOperation2MPIOp(AllreduceOperation op) noexcept {
    switch (op) {
        case CommunicationBackend::ALLREDUCE_OP_SUM:
            return MPI_SUM;
    }
    return MPI_OP_NULL;
}

int MPIBackend::processesImpl_(int *argc, char ***argv) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        initialize_(argc, argv);
    }
    return processes_;
}

int MPIBackend::processRankImpl_(int *argc, char ***argv) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        initialize_(argc, argv);
    }
    return processRank_;
}

void MPIBackend::initialize_(int *argc, char ***argv) {
    int provided, required = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(argc, argv, required, &provided);
    if (provided < required) {
        std::cerr << "ERROR: environment dose not provide mpi thread requirement" << std::endl;
        throw std::runtime_error(
                "environment dose not provide MPI_THREAD_MULTIPLE requirement"
        );
    }
    MPI_Comm_size(MPI_COMM_WORLD, &processes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank_);
    initialized_ = true;
}

int MPIBackend::processes() const {
    return processesImpl_(nullptr, nullptr);
}

int MPIBackend::processRank() const {
    return processRankImpl_(nullptr, nullptr);
}

}}}


