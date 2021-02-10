//
// Created by LYL232 on 2021/2/7.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H
namespace lyl232 { namespace experiment { namespace ddl {
extern "C" {
int mpi_world_size();

int mpi_world_rank();
}
}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_C_API_H
