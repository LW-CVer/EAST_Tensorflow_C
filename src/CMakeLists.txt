aux_source_directory(./ east_src)
add_library(${EAST_LIBRARY_NAME} SHARED ${east_src})

set(OPENCV_NEED_LIBS opencv_imgproc)
target_link_libraries(${EAST_LIBRARY_NAME} PRIVATE ${OPENCV_NEED_LIBS} )
target_link_libraries(${EAST_LIBRARY_NAME} PRIVATE ${TF_LIBS})
