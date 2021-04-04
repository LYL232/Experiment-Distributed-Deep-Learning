//
// Created by LYL232 on 2021/3/21.
//

#include "communicate/backend/mpi/MPICommunicator.h"
#include "communicate/backend/mpi/MPIBackend.h"

namespace lyl232 { namespace experiment { namespace ddl {

MPICommunicator::MPICommunicator(std::shared_ptr<MPI_Comm> mpiComm, int rank, int size) :
        Communicator(rank, size), mpiComm_(std::move(mpiComm)) {}

StatusCode MPICommunicator::allreduce(
        void *sendBuffer, void *recvBuffer,
        size_t elements, DataType dtype,
        AllreduceOperation op) const {
    MPI_Allreduce(
            sendBuffer, recvBuffer,
            (int) elements,
            MPIBackend::DataType2MPIType(dtype),
            AllreduceOperation2MPIOp(op),
            *mpiComm_
    );
    // todo: status check
    return STATUS_OK;
}

StatusCode MPICommunicator::broadcast(
        void *buffer, size_t elements, DataType dtype,
        int rootRank) const {
    MPI_Bcast(
            buffer,
            (int) elements,
            MPIBackend::DataType2MPIType(dtype),
            rootRank,
            *mpiComm_
    );
    // todo: status check
    return STATUS_OK;
}

std::shared_ptr<Communicator> MPICommunicator::split(int color, int key) const {
    auto *newComm = new MPI_Comm;
    MPI_Comm_split(*mpiComm_, color, key, newComm);
    int rank, size;
    MPI_Comm_rank(*newComm, &rank);
    MPI_Comm_size(*newComm, &size);
    auto res = std::make_shared<MPICommunicator>(std::shared_ptr<MPI_Comm>(newComm), rank, size);
    communicatorMap_().emplace(res->id(), res);
    return res;
}

MPI_Comm &MPICommunicator::mpiComm() const {
    return *mpiComm_;
}

Communicator::ID MPICommunicator::id() const noexcept {
    return reinterpret_cast<ID>(mpiComm_.get());
}

MPI_Op MPICommunicator::AllreduceOperation2MPIOp(AllreduceOperation op) noexcept {
    switch (op) {
        case Communicator::ALLREDUCE_OP_SUM:
            return MPI_SUM;
    }
    return MPI_OP_NULL;
}

}}}