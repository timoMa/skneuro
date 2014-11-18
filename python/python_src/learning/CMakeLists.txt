




#--------------------------------------------------------------
# Add MODULES  
#--------------------------------------------------------------
add_library( _learning SHARED  py_learning.cxx
    py_feature_extraction.cxx
    py_compare.cxx
    py_gt_helper.cxx
    py_grf.cxx
)


#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _learning
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)

#--------------------------------------------------------------
# RENAME LIBRARIES
#--------------------------------------------------------------
set_target_properties(_learning PROPERTIES PREFIX "")

#--------------------------------------------------------------
# copy from build to src
#--------------------------------------------------------------
get_property(location TARGET _learning PROPERTY LOCATION)
add_custom_command(TARGET _learning
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/learning
)

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(TARGETS _learning LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/_learning)
install(FILES ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/learning/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/learning)

