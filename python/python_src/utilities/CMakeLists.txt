




#--------------------------------------------------------------
# Add MODULES  
#--------------------------------------------------------------
add_library( _utilities SHARED  py_utilities.cxx
    py_utilities_block.cxx
    py_utilities_blocking.cxx
    py_utilities_block_data.cxx
)




#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _utilities
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)

#--------------------------------------------------------------
# RENAME LIBRARIES
#--------------------------------------------------------------
set_target_properties(_utilities PROPERTIES PREFIX "")

#--------------------------------------------------------------
# copy from build to src
#--------------------------------------------------------------
get_property(location TARGET _utilities PROPERTY LOCATION)
add_custom_command(TARGET _utilities
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/utilities
)

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(TARGETS _utilities LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/_utilities)
install(FILES ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/utilities/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/utilities)

