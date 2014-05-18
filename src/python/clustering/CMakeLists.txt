




#--------------------------------------------------------------
# Add MODULES  
#--------------------------------------------------------------
add_library( _clustering SHARED  py_clustering.cpp
py_mini_batch_k_means.cxx
)




#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _clustering
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)

#--------------------------------------------------------------
# RENAME LIBRARIES
#--------------------------------------------------------------
set_target_properties(_clustering PROPERTIES PREFIX "")

#--------------------------------------------------------------
# copy from build to src
#--------------------------------------------------------------
get_property(location TARGET _clustering PROPERTY LOCATION)
add_custom_command(TARGET _clustering
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}/python_module/skneuro/clustering
)

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(TARGETS _clustering LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/clustering)
install(FILES ${CMAKE_SOURCE_DIR}/python_module/skneuro/clustering/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/clustering)

