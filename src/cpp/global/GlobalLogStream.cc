//
// Created by LYL232 on 2021/2/12.
//

#include "GlobalLogStream.h"

std::ostream &GlobalLogStream::stream() {
    return *stream_;
}

GlobalLogStream::~GlobalLogStream() {
    streamDestructor_();
}
