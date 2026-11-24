#ifndef CSI_LIVE_H
#define CSI_LIVE_H

#include <stdbool.h>
#include <stdint.h>
#include "esp_wifi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Call once from the CSI stream worker, before processing samples.
bool csi_live_init(void);

// Call from the FreeRTOS worker, NEVER the high-priority Wi-Fi callback.
// Returns true if the frame passed the PHY validation gate.
bool csi_live_process(const wifi_pkt_rx_ctrl_t *ctrl,
                      const int8_t *iq, uint16_t len, bool first_word_invalid);

#ifdef __cplusplus
}
#endif
#endif
