add_definitions( -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION )
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
#try to guess the site-packages path
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_CMD OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYTHON_SITE_PACKAGES)
    #if site-packages location was specified on the command line, use that
    #instead
    SET(PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES_CMD} CACHE TYPE "Where to install python modules")
endif()



#--------------------------------------------------------------
# add submodules
#--------------------------------------------------------------
add_subdirectory(core) 
add_subdirectory(clustering) 


#--------------------------------------------------------------
# install main init
#--------------------------------------------------------------
install(FILES ${CMAKE_SOURCE_DIR}/python/python_module/skneuro/__init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/skneuro)

