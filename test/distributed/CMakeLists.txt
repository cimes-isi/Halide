tests(GROUPS correctness_distributed
      SOURCES
      distributed.cpp
      )

target_link_libraries(correctness_distributed_distributed PRIVATE ${MPI_CXX_LIBRARIES})
