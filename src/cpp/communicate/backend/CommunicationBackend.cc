//
// Created by LYL232 on 2021/2/11.
//

#include "communicate/backend/CommunicationBackend.h"
#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::shared_ptr<Communicator> CommunicationBackend::world_ = std::shared_ptr<Communicator>(nullptr);

std::shared_ptr<Communicator> CommunicationBackend::worldCommunicator() const {
    CALLING_ABSTRACT_INTERFACE_ERROR("CommunicationBackend::worldCommunicator()");
}

}}}