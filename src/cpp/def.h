//
// Created by LYL232 on 2021/2/6.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_DEF_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_DEF_H

#include <cstddef>
#include "tensorflow/core/framework/types.h"

namespace lyl232 { namespace experiment { namespace ddl {

typedef unsigned char uchar;

using tensorflow::DataType;
//enum DataType : int {
//    DT_INVALID = 0,
//    DT_FLOAT = 1,
//    DT_DOUBLE = 2,
//    DT_INT32 = 3,
//    DT_UINT8 = 4,
//    DT_INT16 = 5,
//    DT_INT8 = 6,
//    DT_STRING = 7,
//    DT_COMPLEX64 = 8,
//    DT_INT64 = 9,
//    DT_BOOL = 10,
//    DT_QINT8 = 11,
//    DT_QUINT8 = 12,
//    DT_QINT32 = 13,
//    DT_BFLOAT16 = 14,
//    DT_QINT16 = 15,
//    DT_QUINT16 = 16,
//    DT_UINT16 = 17,
//    DT_COMPLEX128 = 18,
//    DT_HALF = 19,
//    DT_RESOURCE = 20,
//    DT_VARIANT = 21,
//    DT_UINT32 = 22,
//    DT_UINT64 = 23,
//    DT_FLOAT_REF = 101,
//    DT_DOUBLE_REF = 102,
//    DT_INT32_REF = 103,
//    DT_UINT8_REF = 104,
//    DT_INT16_REF = 105,
//    DT_INT8_REF = 106,
//    DT_STRING_REF = 107,
//    DT_COMPLEX64_REF = 108,
//    DT_INT64_REF = 109,
//    DT_BOOL_REF = 110,
//    DT_QINT8_REF = 111,
//    DT_QUINT8_REF = 112,
//    DT_QINT32_REF = 113,
//    DT_BFLOAT16_REF = 114,
//    DT_QINT16_REF = 115,
//    DT_QUINT16_REF = 116,
//    DT_UINT16_REF = 117,
//    DT_COMPLEX128_REF = 118,
//    DT_HALF_REF = 119,
//    DT_RESOURCE_REF = 120,
//    DT_VARIANT_REF = 121,
//    DT_UINT32_REF = 122,
//    DT_UINT64_REF = 123,
//    DataType_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::min(),
//    DataType_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::max()
//};

enum StatusCode : uchar {
    STATUS_OK = 0,
    STATUS_MPI_ERROR = 1,
    STATUS_ERROR_UNKNOWN
};

inline tensorflow::Status statusCode2TFStatus(StatusCode code) {
    switch (code) {
        case STATUS_OK:
            return tensorflow::Status::OK();
        case STATUS_MPI_ERROR:
            return tensorflow::errors::Unknown("mpi failed.");
        default:
            return tensorflow::errors::Unknown("unknown reason");
    }
}
}}}
#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_DEF_H
