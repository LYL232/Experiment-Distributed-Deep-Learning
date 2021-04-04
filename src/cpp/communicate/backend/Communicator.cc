//
// Created by LYL232 on 2021/3/21.
//

#include "communicate/backend/Communicator.h"
#include "global/Global.h"
#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

Communicator::Communicator(int rank, int size) : rank_(rank), size_(size) {}


int Communicator::rank() const noexcept {
    return rank_;
}

int Communicator::size() const noexcept {
    return size_;
}

std::shared_ptr<Communicator> Communicator::split(int color, int key) const {
    CALLING_ABSTRACT_INTERFACE_ERROR("Communicator::split(int color, int key)");
}

StatusCode Communicator::allreduce(
        void *sendBuffer, void *recvBuffer, size_t elements, DataType dtype,
        AllreduceOperation op) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "Communicator::allreduce("
            "void *sendBuffer, void *recvBuffer,"
            " size_t elements, DataType dtype, "
            " Operation op)");
}

StatusCode Communicator::broadcast(
        void *buffer, size_t elements, DataType dtype,
        int rootRank) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "Communicator::broadcast("
            "void *buffer,"
            " size_t elements, DataType dtype, "
            " int rootRank)");
}

Communicator::ID Communicator::id() const {
    CALLING_ABSTRACT_INTERFACE_ERROR("Communicator::id()");
}

bool Communicator::operator==(const Communicator &other) const noexcept {
    return id() == other.id();
}

std::map<Communicator::ID, std::shared_ptr<Communicator>> &Communicator::communicatorMap_() noexcept {
    return *Global::get().communicatorMap_;
}

}}}