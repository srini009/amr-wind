target_sources(${amr_wind_lib_name}
  PRIVATE
    ascent.cpp
  )
#target_link_libraries(${amr_wind_lib_name} PRIVATE PkgConfig::ams-client)
#target_include_directories(${amr_wind_lib_name} PUBLIC ${ams-client_INCLUDE_DIRS})
