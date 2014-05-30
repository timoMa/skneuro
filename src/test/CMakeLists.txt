
find_package(Boost COMPONENTS   unit_test_framework REQUIRED)

add_executable( test_accumulator  test_accumulator.cxx )
target_link_libraries(test_accumulator ${Boost_LIBRARIES})
add_test( test_accumulator test_accumulator )