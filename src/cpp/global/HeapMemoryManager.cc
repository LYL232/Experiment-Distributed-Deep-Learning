//
// Created by LYL232 on 2021/6/1.
//

#include <sstream>
#include "global/HeapMemoryManager.h"

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK

#include "global/Global.h"

#endif

namespace lyl232 { namespace experiment { namespace ddl {

HeapMemoryManager::HeapMemoryManager() :
        mutex_(PTHREAD_MUTEX_INITIALIZER),
        numBytesAllocate_(0), numBytesDeallocate_(0),
        usingMemSize_(0), maxUsedMemeSize_(0),
        ptrMemSize_(), typeAllocateNum_(), typeDeallocateNum_(),
        typeNames_() {}


HeapMemoryManager::~HeapMemoryManager() {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK
    using namespace std;
    ostringstream os;
    os << "\nHeap Memory Track:\nmax usage: " << (double) maxUsedMemeSize_ / 1024.0 / 1024.0 << "MB\n";
    os << "bytes array: allocate num: " << numBytesAllocate_ << ", deallocate num: " << numBytesDeallocate_ << "\n";

    os << "type allocation:\n";
    for (auto iter = typeAllocateNum_.begin(); iter != typeAllocateNum_.end(); ++iter) {
        auto nameIter = typeNames_.find(iter->first);
        if (nameIter == typeNames_.end()) {
            os << "Not Named: ";
        } else {
            os << nameIter->second << ": ";
        }
        os << "allocate num: " << iter->second << ", deallocate num: ";
        auto deallocNumIter = typeDeallocateNum_.find(iter->first);
        if (deallocNumIter == typeDeallocateNum_.end()) {
            os << "0\n";
        } else {
            os << deallocNumIter->second << "\n";
        }
    }
    GLOBAL_INFO(os.str())
#endif
    pthread_mutex_destroy(&mutex_);
}

void *HeapMemoryManager::allocateBytes(size_t bytes) const {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK
    auto *ptr = new char[bytes];
    // todo: ptr申请失败会为nullptr, 需要作出相应检查
    pthread_mutex_lock(&mutex_);
    numBytesAllocate_ += 1;
    ptrMemSize_[(size_t) ptr] = bytes;
    usingMemSize_ += bytes;
    if (usingMemSize_ > maxUsedMemeSize_) {
        maxUsedMemeSize_ = usingMemSize_;
    }
    pthread_mutex_unlock(&mutex_);
    return ptr;
#else
    return new char[bytes];
#endif
}

void HeapMemoryManager::deallocateBytes(void *ptr) const {
    if (ptr == nullptr) {
        return;
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK
    pthread_mutex_lock(&mutex_);
    numBytesDeallocate_ += 1;
    usingMemSize_ -= ptrMemSize_[(size_t) ptr];
    ptrMemSize_.erase((size_t) ptr);
    pthread_mutex_unlock(&mutex_);
#endif
    delete[](char *) ptr;
}

//template<typename T, typename... Args>
//T *HeapMemoryManager::allocateType(Args... args) {
//#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK
//    auto *ptr = new T(args...);
//    // todo: ptr申请失败会为nullptr, 需要作出相应检查
//    size_t bytes = sizeof(T), typeId = typeid(T).hash_code();
//    pthread_mutex_lock(&mutex_);
//    ptrMemSize_[(size_t) ptr] = bytes;
//
//    // 记录对应类型的申请次数+1
//    auto allocateNumIter = typeAllocateNum_.find(typeId);
//    if (allocateNumIter == typeAllocateNum_.end()) {
//        typeAllocateNum_[typeId] = 1;
//    } else {
//        allocateNumIter->second += 1;
//    }
//
//    // 注册名字, 如果没有的话
//    auto typeNameIter = typeNames_.find(typeId);
//    if (typeNameIter == typeNames_.end()) {
//        typeNames_[typeId] = typeid(T).name();
//    }
//
//    usingMemSize_ += bytes;
//    if (usingMemSize_ > maxUsedMemeSize_) {
//        maxUsedMemeSize_ = usingMemSize_;
//    }
//    pthread_mutex_unlock(&mutex_);
//    return ptr;
//#else
//    return new T(args...);
//#endif
//}
//
//template<typename T>
//void HeapMemoryManager::deallocateType(T *ptr) {
//#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK
//    if (ptr == nullptr) {
//        return;
//    }
//    pthread_mutex_lock(&mutex_);
//    size_t typeId = (size_t) ptr;
//    // 记录对应类型的申请次数+1
//    auto deallocateNumIter = typeDeallocateNum_.find(typeId);
//    if (deallocateNumIter == typeDeallocateNum_.end()) {
//        typeDeallocateNum_[typeId] = 1;
//    } else {
//        deallocateNumIter->second += 1;
//    }
//
//    // 注册名字, 如果没有的话
//    auto typeNameIter = typeNames_.find(typeId);
//    if (typeNameIter == typeNames_.end()) {
//        typeNames_[typeId] = typeid(T).name();
//    }
//
//    usingMemSize_ -= ptrMemSize_[typeId];
//    pthread_mutex_unlock(&mutex_);
//#endif
//    delete ptr;
//}

void *HeapMemoryManager::trackAllocateType(void *ptr, const std::type_info &typeInfo, size_t bytes) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK
    size_t typeId = typeInfo.hash_code();
    pthread_mutex_lock(&mutex_);
    ptrMemSize_[(size_t) ptr] = bytes;

    // 记录对应类型的申请次数+1
    auto allocateNumIter = typeAllocateNum_.find(typeId);
    if (allocateNumIter == typeAllocateNum_.end()) {
        typeAllocateNum_[typeId] = 1;
    } else {
        allocateNumIter->second += 1;
    }

    // 注册名字, 如果没有的话
    auto typeNameIter = typeNames_.find(typeId);
    if (typeNameIter == typeNames_.end()) {
        typeNames_[typeId] = typeInfo.name();
    }

    usingMemSize_ += bytes;
    if (usingMemSize_ > maxUsedMemeSize_) {
        maxUsedMemeSize_ = usingMemSize_;
    }
    pthread_mutex_unlock(&mutex_);
#endif
    return ptr;
}

void HeapMemoryManager::trackDeallocateType(void *ptr, const std::type_info &typeInfo) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_HEAP_MEMORY_TRACK
    pthread_mutex_lock(&mutex_);
    size_t typeId = typeInfo.hash_code(), ptrId = (size_t) ptr;
    // 记录对应类型的申请次数+1
    auto deallocateNumIter = typeDeallocateNum_.find(typeId);
    if (deallocateNumIter == typeDeallocateNum_.end()) {
        typeDeallocateNum_[typeId] = 1;
    } else {
        deallocateNumIter->second += 1;
    }

    // 注册名字, 如果没有的话
    auto typeNameIter = typeNames_.find(typeId);
    if (typeNameIter == typeNames_.end()) {
        typeNames_[typeId] = typeInfo.name();
    }

    usingMemSize_ -= ptrMemSize_[ptrId];
    ptrMemSize_.erase(ptrId);
    pthread_mutex_unlock(&mutex_);
#endif
}

}}}