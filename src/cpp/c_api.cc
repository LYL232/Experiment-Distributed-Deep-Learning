//
// Created by LYL232 on 2021/2/7.
//
#include "global/Global.h"
#include "c_api.h"

namespace lyl232 { namespace experiment { namespace ddl {
int processes() {
    return Global::get().processes();
}

int process_rank() {
    return Global::get().processRank();
}
}}}