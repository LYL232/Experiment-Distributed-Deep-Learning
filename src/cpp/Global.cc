//
// Created by LYL232 on 2021/2/5.
//

#include "mpi.h"
#include <fstream>
#include "Global.h"
#include "communicate/tensor/allreduce/rta/RingTokenAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl {

Global Global::instance_;

Global::~Global() {
    GLOBAL_INFO_WITH_RANK("Global finalizing");
    // 之所以使用指针来指向, 是因为需要在MPI_Finalize之前确认其他的MPI资源得到释放, 而
    // delete指针可以做到这一点
    delete controller_;
    pthread_rwlock_destroy(&rwlock_);
    MPI_Finalize();
    GLOBAL_INFO_WITH_RANK("Global finalized");
    logStreamDestructor_();
    for (auto iter = threadLogStream_.begin(); iter != threadLogStream_.end(); ++iter) {
        delete iter->second;
    }
}

void Global::init() {
    int provided, required = MPI_THREAD_MULTIPLE, rank, size;
    MPI_Init_thread(nullptr, nullptr, required, &provided);
    if (provided < required) {
        std::cerr << "ERROR: mpi environment dose not provide mpi thread requirement" << std::endl;
        throw std::runtime_error(
                "mpi environment dose not provide MPI_THREAD_MULTIPLE requirement"
        );
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    instance_.MPIWorldRank_ = rank;
    instance_.MPIWorldSize_ = size;

    std::ostringstream strStream;
    strStream << "log-" << rank << ".txt";
    auto *log = new std::ofstream(strStream.str());
    instance_.log_ = log;
    instance_.logStreamDestructor_ = [log]() {
        log->close();
        delete log;
    };

    auto *controller = new tensorsallreduce::rta::RingTokenAllreduceController;
    GLOBAL_INFO_WITH_RANK_THREAD_ID("new controller, waiting for initialization");
    while (!controller->initialized()) {
        GLOBAL_INFO_WITH_RANK_THREAD_ID("controller initialization check failed");
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    instance_.controller_ = controller;
    instance_.initialized = true;
    GLOBAL_INFO_WITH_RANK("Global initialized");
}

void Global::log(std::ostringstream &stream, bool clear) const {
    pthread_rwlock_wrlock(&rwlock_);
    *log_ << stream.str() << std::endl;
    pthread_rwlock_unlock(&rwlock_);
    if (clear) {
        stream.str(std::string());
        stream.clear();
    }
}

std::ostringstream &Global::thisThreadLogStream() const {
    std::ostringstream *ptr;
    pthread_rwlock_rdlock(&rwlock_);
    auto iter = threadLogStream_.find(std::this_thread::get_id());
    if (iter == threadLogStream_.end()) {
        pthread_rwlock_unlock(&rwlock_);
        pthread_rwlock_wrlock(&rwlock_);
        ptr = new std::ostringstream();
        threadLogStream_.emplace(std::this_thread::get_id(), ptr);
    } else {
        ptr = iter->second;
    }
    pthread_rwlock_unlock(&rwlock_);
    return *ptr;
}

const Global &Global::get() {
    return instance_;
}

}}}