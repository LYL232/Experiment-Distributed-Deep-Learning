//
// Created by LYL232 on 2021/2/6.
//

#include "mpi.h"
#include "Global.h"
#include "communicate/communication/MPICommunication.h"

namespace lyl232 { namespace experiment { namespace ddl {
using namespace tensorflow;

int MPICommunication::DataType2MPIType(DataType dtype) {
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
    auto &global = Global::get();
    std::ostringstream strStream;
    strStream << "trying getting unsupported DataType: " << dtype;
    global.log(strStream);
    return MPI_DATATYPE_NULL;
}

}}}
