




#--------------------------------------------------------------
# Add MODULES  
#--------------------------------------------------------------
add_library( _oversegmentation SHARED  py_oversegmentation.cxx
    py_oversegmentation.cxx
    py_seg_helper.cxx
)


#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _oversegmentation
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)

#--------------------------------------------------------------
# RENAME LIBRARIES
#--------------------------------------------------------------
set_target_properties(_oversegmentation PROPERTIES PREFIX "")

#--------------------------------------------------------------
# copy from build to src
#--------------------------------------------------------------
get_property(location TARGET _oversegmentation PROPERTY LOCATION)
add_custom_command(TARGET _oversegmentation
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/oversegmentation
)

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(TARGETS _oversegmentation LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/_oversegmentation)
install(FILES ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/oversegmentation/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/oversegmentation)

