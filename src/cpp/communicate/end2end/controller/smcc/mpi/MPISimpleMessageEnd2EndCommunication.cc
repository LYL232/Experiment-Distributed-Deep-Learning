//
// Created by LYL232 on 2021/2/18.
//

#include "mpi.h"
#include "communicate/end2end/controller/smcc/mpi/MPISimpleMessageEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace smcc {

StatusCode MPISimpleMessageEnd2EndCommunication::sendMessage(const Message &message) {
    return SimpleMessageEnd2EndCommunication::sendMessage(message);
}

SimpleMessageEnd2EndCommunication::SharedMessage MPISimpleMessageEnd2EndCommunication::listen() {
//    MPI_Recv(MPI_ANY_SOURCE);
    return SimpleMessageEnd2EndCommunication::listen();
}

}}}}