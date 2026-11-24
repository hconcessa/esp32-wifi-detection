#ifndef CSI_STREAM_H
#define CSI_STREAM_H

#include "esp_wifi.h"

// Start the CSI output worker before enabling the Wi-Fi CSI callback.
void csi_stream_start(void);

// Called from the Wi-Fi task, not from an ISR. Copies CSI data without blocking.
void csi_stream_submit(const wifi_csi_info_t *info);

#endif
