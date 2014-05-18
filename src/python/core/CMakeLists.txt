




#--------------------------------------------------------------
# Add MODULES  
#--------------------------------------------------------------
add_library( _core SHARED  py_core.cpp
py_core_a.cxx
)




#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _core
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)

#--------------------------------------------------------------
# RENAME LIBRARIES
#--------------------------------------------------------------
set_target_properties(_core PROPERTIES PREFIX "")

#--------------------------------------------------------------
# copy from build to src
#--------------------------------------------------------------
get_property(location TARGET _core PROPERTY LOCATION)
add_custom_command(TARGET _core
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}/python_module/skneuro/core
)

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(TARGETS _core LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/core)
install(FILES ${CMAKE_SOURCE_DIR}/python_module/skneuro/core/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/core)

