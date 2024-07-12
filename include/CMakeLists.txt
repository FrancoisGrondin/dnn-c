#Scan for headers and add them for installation
file(GLOB_RECURSE DNNC_ALL_INCLUDES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

message(STATUS "HEADER FILES: ${DNNC_ALL_INCLUDES}")

add_custom_target(dnnc_headers SOURCES ${DNNC_ALL_INCLUDES})

install(FILES ${DNNC_ALL_INCLUDES} DESTINATION include)