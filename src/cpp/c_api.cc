//
// Created by LYL232 on 2021/2/7.
//
#include "Global.h"
#include "c_api.h"

namespace lyl232 { namespace experiment { namespace ddl {
int mpi_world_size() {
    return Global::get().MPIWorldSize();
}

int mpi_world_rank() {
    return Global::get().MPIWorldRank();
}
}}}