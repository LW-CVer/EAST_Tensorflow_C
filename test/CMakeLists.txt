add_executable(test_east test.cpp)
target_link_libraries(test_east ${EAST_LIBRARY_NAME})
set(OPENCV_NEED_LIBS opencv_imgproc opencv_highgui)
target_link_libraries(test_east ${OPENCV_NEED_LIBS})
