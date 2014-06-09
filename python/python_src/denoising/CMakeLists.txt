




#--------------------------------------------------------------
# Add MODULES  
#--------------------------------------------------------------
add_library( _denoising SHARED  py_denoising.cxx
    py_denoising.cxx
    py_denoising_non_local_mean.cxx
    py_denoising_ball_rank_order.cxx
)


#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _denoising
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)

#--------------------------------------------------------------
# RENAME LIBRARIES
#--------------------------------------------------------------
set_target_properties(_denoising PROPERTIES PREFIX "")

#--------------------------------------------------------------
# copy from build to src
#--------------------------------------------------------------
get_property(location TARGET _denoising PROPERTY LOCATION)
add_custom_command(TARGET _denoising
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/denoising
)

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(TARGETS _denoising LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/_denoising)
install(FILES ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/denoising/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro/denoising)

