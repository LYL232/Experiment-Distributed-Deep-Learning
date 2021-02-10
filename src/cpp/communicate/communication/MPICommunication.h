//
// Created by LYL232 on 2021/2/6.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPICOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPICOMMUNICATION_H

#include "def.h"
#include "communicate/communication/Communication.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MPICommunication : virtual public Communication {
public:
    enum MPICommunicateTag : int {
        MPI_TAG_RTA_COMMUNICATE_META = 0,
        MPI_TAG_RTA_COMMUNICATE_MSG = 1
    };

    /**
     * DataType到MPI_TYPE的映射
     * @param dtype
     * @return MPI_TYPE
     */
    static int DataType2MPIType(DataType dtype);
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPICOMMUNICATION_H
