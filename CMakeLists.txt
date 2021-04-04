cmake_minimum_required(VERSION 3.10)
project(ddl)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 11)

list(APPEND SOURCES
        "${PROJECT_SOURCE_DIR}/src/cpp/global/Global.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/global/GlobalLog.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/global/initialize.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/c_api.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/op/AllreduceOp.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/op/BroadcastOp.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/op/ForwardAndSendOp.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/op/ReceiveTensorOp.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/op/SendTensorOp.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/backend/Communicator.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/backend/CommunicationBackend.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/backend/mpi/MPIBackend.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/backend/mpi/MPICommunicator.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/TensorCommunicateRequest.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/controller/rtc/RingTokenCommunicateController.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/controller/rtc/RingTokenCommunicateHandler.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/controller/rtc/Token.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/controller/rtc/RingTokenCommunication.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/controller/rtc/mpi/MPIRingTokenCommunication.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/controller/rtc/mpi/MPIRingTokenCommunicateController.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/TensorCollectiveCommunicateRequest.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/broadcast/TensorBroadcastRequest.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/collective/allreduce/TensorAllreduceRequest.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/end2end/TensorEnd2EndCommunicateRequest.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/end2end/send/TensorSendCommunicateRequest.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/end2end/receive/TensorReceiveCommunicateRequest.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/end2end/controller/TensorEnd2EndCommunicateController.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/end2end/controller/bcc/BlockedEnd2EndCommunicateController.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/end2end/controller/bcc/BlockedEnd2EndCommunication.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/tensor/end2end/controller/bcc/mpi/MPIBlockedEnd2EndCommunication.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/message/MessageController.cc"
        "${PROJECT_SOURCE_DIR}/src/cpp/communicate/message/mpi/MPIMessageController.cc"
        )

# PYTHON
if (NOT PYTHON_EXECUTABLE)
    if (DEFINED ENV{PYTHON_EXECUTABLE})
        set(PY_EXE $ENV{PYTHON_EXECUTABLE})
    else ()
        find_package(Python 3.6 COMPONENTS Interpreter REQUIRED)
        set(PY_EXE ${Python_EXECUTABLE})
    endif ()
else ()
    set(PY_EXE ${PYTHON_EXECUTABLE})
endif ()
message(STATUS "Using command ${PY_EXE}")

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "current platform: Windows")
    if (DEFINED ENV{MPI_INCLUDE_PATH})
        message(STATUS "found MPI_INCLUDE_PATH: $ENV{MPI_INCLUDE_PATH}")
        include_directories($ENV{MPI_INCLUDE_PATH})
    else ()
        message(FATAL_ERROR "error: can not find MPI_INCLUDE_PATH")
    endif ()
else ()
    message(STATUS "not Windows platform")
    find_package(MPI REQUIRED)
    if (MPI_FOUND)
        include_directories(SYSTEM ${MPI_INCLUDE_PATH})
        list(APPEND LINKER_LIBS ${MPI_LIBRARIES})
    else ()
        message(FATAL_ERROR "error: can not find MPI")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif ()

execute_process(COMMAND ${PY_EXE} -c "import tensorflow as tf; print(tf.sysconfig.get_include() or [' ']); print(' '.join(tf.sysconfig.get_link_flags() or [' '])); print(' '.join(tf.sysconfig.get_compile_flags() or [' ']))"
        OUTPUT_VARIABLE FIND_TENSORFLOW_OUTPUTS
        ERROR_VARIABLE FIND_TENSORFLOW_ERRORS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE "\n" ";" FIND_TENSORFLOW_OUTPUTS "${FIND_TENSORFLOW_OUTPUTS}")

list(LENGTH FIND_TENSORFLOW_OUTPUTS LEN)
if (NOT LEN EQUAL "3")
    message(FATAL_ERROR "error when trying to get tensorflow information, abort.
     error message: ${FIND_TENSORFLOW_ERRORS}")
endif ()

list(GET FIND_TENSORFLOW_OUTPUTS 0 TENSORFLOW_INCLUDE)
list(GET FIND_TENSORFLOW_OUTPUTS 1 TENSORFLOW_LIBRARIES)
string(REPLACE " " ";" TENSORFLOW_LIBRARIES "${TENSORFLOW_LIBRARIES}")
list(GET FIND_TENSORFLOW_OUTPUTS 2 TENSORFLOW_COMPILE_FLAGS)
if ("${TENSORFLOW_COMPILE_FLAGS}" MATCHES "-D_GLIBCXX_USE_CXX11_ABI=1")
    set(TENSORFLOW_CXX11 TRUE)
else ()
    set(TENSORFLOW_CXX11 FALSE)
endif ()

message(STATUS TENSORFLOW_INCLUDE:${TENSORFLOW_INCLUDE})
message(STATUS TENSORFLOW_LIBRARIES:${TENSORFLOW_LIBRARIES})
message(STATUS TENSORFLOW_COMPILE_FLAGS:${TENSORFLOW_COMPILE_FLAGS})
message(STATUS TENSORFLOW_CXX11:${TENSORFLOW_CXX11})

set(CMAKE_CXX_FLAGS "${TENSORFLOW_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS} -O2 -Wall")

#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LINK_FLAGS}")

message(STATUS CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS})

include_directories(${PROJECT_SOURCE_DIR}/src/cpp)

add_library(ddl SHARED ${SOURCES})
target_link_libraries(ddl ${LINKER_LIBS} ${TENSORFLOW_LIBRARIES})
set_target_properties(ddl PROPERTIES SUFFIX ".so")
set_target_properties(ddl PROPERTIES PREFIX "")
set_target_properties(ddl PROPERTIES OUTPUT_NAME "lib")
