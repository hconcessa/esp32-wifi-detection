cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(esp32_wifi_detection)

# Os modelos usam somente FULLY_CONNECTED e LOGISTIC. O componente esp-tflite-micro
# compila todos os kernels, e svdf_common.cc provoca "internal compiler error (RTL pass ira)"
# em algumas toolchains Windows. Remover SVDF da biblioteca resolve sem tocar em
# managed_components (que é recriado pelo gerenciador de componentes).
if(TARGET __idf_espressif__esp-tflite-micro)
    get_target_property(_tflm_srcs __idf_espressif__esp-tflite-micro SOURCES)
    list(FILTER _tflm_srcs EXCLUDE REGEX "/kernels/svdf(_common)?\\.cc$")
    set_property(TARGET __idf_espressif__esp-tflite-micro PROPERTY SOURCES ${_tflm_srcs})
    message(STATUS "esp-tflite-micro: kernels SVDF excluídos (modelo usa só FULLY_CONNECTED + LOGISTIC)")
endif()
