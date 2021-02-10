//
// Created by LYL232 on 2021/2/10.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIGLOBAL_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIGLOBAL_H

#include <atomic>
#include "global/Global.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MPIGlobal : public Global {

    MPIGlobal(const MPIGlobal &) = delete;

    MPIGlobal(MPIGlobal &&) = delete;

private:
    static MPIGlobal instance_;

    MPIGlobal();

    void init();
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIGLOBAL_H
