//
// Created by LYL232 on 2021/3/21.
//

#include "communicate/backend/mpi/MPICommunicator.h"
#include "communicate/backend/mpi/MPIBackend.h"
#include "global/Global.h"

namespace lyl232 { namespace experiment { namespace ddl {

MPICommunicator::MPICommunicator(std::shared_ptr<MPI_Comm> mpiComm, int rank, int size) :
        Communicator(rank, size), mpiComm_(std::move(mpiComm)) {}

StatusCode MPICommunicator::allreduce(
        void *sendBuffer, void *recvBuffer,
        size_t elements, DataType dtype,
        AllreduceOperation op) const {
    // todo: 分批次发送 if elements > max_int
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

StatusCode
MPICommunicator::allgather(
        void *sendBuffer, size_t elements,
        void *recvBuffer,
        const std::vector<size_t> &recvCounts,
        const std::vector<size_t> &displs,
        DataType dtype) const {
    // todo: 分批次发送 if elements > max_int
    // todo: 想想处理recvCounts 和 displs 的size_t > max_int 时的处理方法
    assert(recvCounts.size() == (size_t) size());
    assert(displs.size() == (size_t) size());
    std::vector<int> intRecvCounts, intDispls;
    intRecvCounts.resize(size());
    intDispls.resize(size());

    for (int i = 0; i < size(); ++i) {
        intRecvCounts[i] = (int) recvCounts[i];
        intDispls[i] = (int) displs[i];
    }

    MPI_Allgatherv(
            sendBuffer, (int) elements,
            MPIBackend::DataType2MPIType(dtype),
            recvBuffer, intRecvCounts.data(),
            intDispls.data(), MPIBackend::DataType2MPIType(dtype),
            *mpiComm_
    );
    // todo: status check
    return STATUS_OK;
}

StatusCode
MPICommunicator::allgather(
        void *sendBuffer, size_t sendElements, void *recvBuffer, size_t recvElements,
        DataType dtype) const {
    // todo: 分批次发送 if elements > max_int
    MPI_Allgather(
            sendBuffer, (int) sendElements,
            MPIBackend::DataType2MPIType(dtype),
            recvBuffer, (int) recvElements,
            MPIBackend::DataType2MPIType(dtype),
            *mpiComm_
    );
    // todo: status check
    return STATUS_OK;
}

StatusCode MPICommunicator::broadcast(
        void *buffer, size_t elements, DataType dtype,
        int rootRank) const {
    // todo: 分批次发送 if elements > max_int
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
    auto *newComm = TRACK_TYPE_ALLOCATE(memManager_, new MPI_Comm, MPI_Comm);
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