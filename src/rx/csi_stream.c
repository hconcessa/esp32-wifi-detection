// Fila entre o callback Wi-Fi CSI (alta prioridade) e a tarefa de inferência.
// O callback só copia o pacote (sem bloquear); todo o processamento roda em csi_stream_task.
#include "csi_stream.h"
#include "csi_live.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#define CSI_QUEUE_LENGTH 16
#define CSI_MAX_DATA_BYTES 512

static const char *TAG = "csi_stream";

typedef struct {
    uint32_t sequence;
    wifi_pkt_rx_ctrl_t rx_ctrl;
    uint16_t len;
    bool first_word_invalid;
    int8_t data[CSI_MAX_DATA_BYTES];
} csi_record_t;

static QueueHandle_t s_queue = NULL;
static volatile uint32_t s_received = 0;
static volatile uint32_t s_queue_drops = 0;
static volatile uint32_t s_invalid = 0;
static volatile uint32_t s_sequence_missing = 0;
static volatile uint32_t s_processed = 0;
static volatile uint32_t s_unsupported = 0;
static uint32_t s_last_sequence = 0;
static bool s_have_last_sequence = false;

static void csi_stream_task(void *arg)
{
    (void)arg;
    if (!csi_live_init()) {
        ESP_LOGE(TAG, "Falha ao inicializar os modelos (veja CSI_LIVE acima)");
        vTaskDelete(NULL);
        return;
    }
    static csi_record_t record;
    TickType_t last_report = xTaskGetTickCount();
    while (true) {
        if (xQueueReceive(s_queue, &record, pdMS_TO_TICKS(100)) == pdTRUE) {
            if (csi_live_process(&record.rx_ctrl, record.data, record.len, record.first_word_invalid)) {
                s_processed++;
            } else {
                s_unsupported++;
            }
        }
        if (xTaskGetTickCount() - last_report >= pdMS_TO_TICKS(10000)) {
            last_report = xTaskGetTickCount();
            // queue_drops: a tarefa não deu conta; sequence_missing: pacotes perdidos no rádio
            ESP_LOGI(TAG, "CSI_STATS received=%" PRIu32 " processed=%" PRIu32 " unsupported=%" PRIu32
                     " queue_drops=%" PRIu32 " sequence_missing=%" PRIu32 " invalid=%" PRIu32 " queue_pending=%u",
                     s_received, s_processed, s_unsupported, s_queue_drops, s_sequence_missing, s_invalid,
                     (unsigned)uxQueueMessagesWaiting(s_queue));
        }
    }
}

void csi_stream_start(void)
{
    s_queue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(csi_record_t));
    configASSERT(s_queue != NULL);
    BaseType_t created = xTaskCreatePinnedToCore(csi_stream_task, "csi_stream", 8192, NULL, 4, NULL, 1);
    configASSERT(created == pdPASS);
}

void csi_stream_submit(const wifi_csi_info_t *info)
{
    if (!info || !info->buf || !info->payload || !s_queue || info->len == 0 || info->len > CSI_MAX_DATA_BYTES) {
        s_invalid++;
        return;
    }
    uint32_t sequence;  // contador enviado pelo TX no payload ESP-NOW
    memcpy(&sequence, info->payload + 15, sizeof(sequence));
    if (s_have_last_sequence) {
        const uint32_t gap = sequence - s_last_sequence;
        if (gap > 1 && gap < 10000) s_sequence_missing += gap - 1;
    }
    s_last_sequence = sequence;
    s_have_last_sequence = true;
    s_received++;

    csi_record_t record = {
        .sequence = sequence,
        .rx_ctrl = info->rx_ctrl,
        .len = info->len,
        .first_word_invalid = info->first_word_invalid,
    };
    memcpy(record.data, info->buf, info->len);
    if (xQueueSend(s_queue, &record, 0) != pdTRUE) s_queue_drops++;
}
