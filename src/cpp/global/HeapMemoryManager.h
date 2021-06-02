//
// Created by LYL232 on 2021/6/1.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MEMORY_MANAGER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MEMORY_MANAGER_H

#include <cstddef>
#include <pthread.h>
#include <unordered_map>
#include "def.h"
#include "global/LogConfig.h"

namespace lyl232 { namespace experiment { namespace ddl {

/**
 * 堆内存管理类, 主要目的是追踪cpp端的高频内存申请和释放, 对于那些只在初始化和结束时的内存申请和释放, 该类不负责
 */
class HeapMemoryManager {
public:
    HeapMemoryManager();

    void *allocateBytes(size_t bytes) const;

    void deallocateBytes(void *ptr) const;

//    tensorflow的加载库机制好像识别不了带模板的方法, 会出现undefined symbol错误, 暂时放弃以下实现
//
//    template<typename T, typename ...Args>
//    T *allocateType(Args ... args);
//
//    template<typename T>
//    void deallocateType(T *ptr);

    void *trackAllocateType(void *ptr, const std::type_info &typeInfo, size_t bytes);

    void trackDeallocateType(void *ptr, const std::type_info &typeInfo);

    ~HeapMemoryManager();

private:
    mutable pthread_mutex_t mutex_;
    mutable size_t numBytesAllocate_, numBytesDeallocate_, usingMemSize_, maxUsedMemeSize_;
    mutable std::unordered_map<size_t, size_t> ptrMemSize_, typeAllocateNum_, typeDeallocateNum_;
    mutable std::unordered_map<size_t, std::string> typeNames_;
};

}}}

#define TRACK_TYPE_ALLOCATE(manager, ptr, type) (type*)(manager->trackAllocateType((ptr), typeid(type), sizeof(type)))
#define TRACK_TYPE_DEALLOCATE(manager, ptr, type) manager->trackDeallocateType(ptr, typeid(type));delete (type*)ptr


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MEMORY_MANAGER_H
