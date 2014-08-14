




#--------------------------------------------------------------
# Add MODULES  
#--------------------------------------------------------------
add_library( _parallel SHARED  py_parallel.cxx
    py_parallel_numpy.cxx
)


#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _parallel
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)

#--------------------------------------------------------------
# RENAME LIBRARIES
#--------------------------------------------------------------
set_target_properties(_parallel PROPERTIES PREFIX "")

#--------------------------------------------------------------
# copy from build to src
#--------------------------------------------------------------
get_property(location TARGET _parallel PROPERTY LOCATION)
add_custom_command(TARGET _parallel
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/parallel
)

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(TARGETS _parallel LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/_parallel)
install(FILES ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/paralell/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/paralell)

