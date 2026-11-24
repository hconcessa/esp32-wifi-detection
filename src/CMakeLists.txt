# Um único diretório src/ com dois firmwares; o alvo escolhe qual compilar:
#   env:tx -> ESP32-C3 transmissor (ESP-NOW 40 Hz)
#   env:rx -> ESP32-WROOM receptor (CSI + 2 modelos TFLite Micro + LED de 3 estados)
if(IDF_TARGET STREQUAL "esp32c3")
    set(APP_SOURCE "tx/app_main.c")
    set(APP_EXTRA_REQUIRES "")
elseif(IDF_TARGET STREQUAL "esp32")
    set(APP_SOURCE
        "rx/app_main.c"
        "rx/csi_stream.c"
        "rx/csi_features.cpp"
        "rx/csi_live.cpp"
        "rx/model_motion_data.cpp"
        "rx/model_presence_data.cpp"
    )
    set(APP_EXTRA_REQUIRES "esp-tflite-micro" esp_driver_gpio esp_timer)
else()
    message(FATAL_ERROR "Alvo não suportado: ${IDF_TARGET}")
endif()

idf_component_register(
    SRCS ${APP_SOURCE}
    INCLUDE_DIRS "." "rx"
    PRIV_REQUIRES
        esp_wifi
        esp_netif
        esp_event
        esp_hw_support
        nvs_flash
        esp_csi_gain_ctrl
        ${APP_EXTRA_REQUIRES}
)
