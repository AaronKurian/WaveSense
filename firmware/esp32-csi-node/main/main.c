#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#define CSI_MAGIC 0x31534357u
#define CSI_VERSION 1u
#define CSI_HEADER_LEN 28u
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1
#define MAX_WIFI_RETRIES 10
#define MAX_CSI_PACKET 1500

static const char *TAG = "wifi-csi-node";
static EventGroupHandle_t s_wifi_event_group;
static int s_wifi_retries;
static int s_udp_sock = -1;
static struct sockaddr_in s_udp_dest;
static uint32_t s_sequence;
static int64_t s_last_send_us;
static uint32_t s_csi_callbacks;
static uint32_t s_csi_sent;
static uint32_t s_csi_send_failures;

static void put_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)(value & 0xffu);
    dst[1] = (uint8_t)(value >> 8);
}

static void put_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t)(value & 0xffu);
    dst[1] = (uint8_t)((value >> 8) & 0xffu);
    dst[2] = (uint8_t)((value >> 16) & 0xffu);
    dst[3] = (uint8_t)((value >> 24) & 0xffu);
}

static void put_u64_le(uint8_t *dst, uint64_t value)
{
    for (int i = 0; i < 8; i++) {
        dst[i] = (uint8_t)((value >> (8 * i)) & 0xffu);
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    (void)arg;
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_wifi_retries < MAX_WIFI_RETRIES) {
            s_wifi_retries++;
            ESP_LOGW(TAG, "Wi-Fi disconnected; retry %d/%d", s_wifi_retries, MAX_WIFI_RETRIES);
            esp_wifi_connect();
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        uint8_t primary = 0;
        wifi_second_chan_t second = WIFI_SECOND_CHAN_NONE;
        if (esp_wifi_get_channel(&primary, &second) == ESP_OK) {
            ESP_LOGI(TAG, "Connected channel: primary=%u second=%d", (unsigned)primary, (int)second);
            if (CONFIG_CSI_WIFI_CHANNEL != 0 && primary != CONFIG_CSI_WIFI_CHANNEL) {
                ESP_LOGW(
                    TAG,
                    "Expected channel %d but AP is on %u; following AP channel",
                    CONFIG_CSI_WIFI_CHANNEL,
                    (unsigned)primary);
            }
        }
        s_wifi_retries = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_sta(void)
{
    if (strlen(CONFIG_CSI_WIFI_SSID) == 0) {
        ESP_LOGE(TAG, "CONFIG_CSI_WIFI_SSID is empty; set it in menuconfig before flashing");
    }

    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {0};
    strlcpy((char *)wifi_config.sta.ssid, CONFIG_CSI_WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strlcpy((char *)wifi_config.sta.password, CONFIG_CSI_WIFI_PASSWORD, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = strlen(CONFIG_CSI_WIFI_PASSWORD) > 0
        ? WIFI_AUTH_WPA_PSK
        : WIFI_AUTH_OPEN;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE,
        pdFALSE,
        portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to SSID: %s", CONFIG_CSI_WIFI_SSID);
    } else {
        ESP_LOGE(TAG, "Failed to connect to SSID: %s", CONFIG_CSI_WIFI_SSID);
    }
}

static void udp_init(void)
{
    if (strcmp(CONFIG_CSI_TARGET_IP, "0.0.0.0") == 0 || strlen(CONFIG_CSI_TARGET_IP) == 0) {
        ESP_LOGE(TAG, "CONFIG_CSI_TARGET_IP is not configured; set laptop IPv4 address before flashing");
        return;
    }

    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "socket failed: errno %d", errno);
        return;
    }

    memset(&s_udp_dest, 0, sizeof(s_udp_dest));
    s_udp_dest.sin_family = AF_INET;
    s_udp_dest.sin_port = htons(CONFIG_CSI_TARGET_PORT);
    if (inet_pton(AF_INET, CONFIG_CSI_TARGET_IP, &s_udp_dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "invalid target IP: %s", CONFIG_CSI_TARGET_IP);
        close(s_udp_sock);
        s_udp_sock = -1;
        return;
    }

    ESP_LOGI(TAG, "UDP target: %s:%d", CONFIG_CSI_TARGET_IP, CONFIG_CSI_TARGET_PORT);
}

static size_t build_csi_packet(const wifi_csi_info_t *info, uint8_t *out, size_t out_len)
{
    if (!info || !info->buf || out_len < CSI_HEADER_LEN) {
        return 0;
    }
    uint16_t csi_len = (uint16_t)info->len;
    if ((CSI_HEADER_LEN + csi_len) > out_len || (csi_len % 2) != 0) {
        return 0;
    }

    uint16_t subcarrier_count = csi_len / 2u;
    put_u32_le(&out[0], CSI_MAGIC);
    out[4] = CSI_VERSION;
    out[5] = CSI_HEADER_LEN;
    out[6] = (uint8_t)CONFIG_CSI_NODE_ID;
    out[7] = info->rx_ctrl.channel;
    put_u32_le(&out[8], s_sequence++);
    put_u64_le(&out[12], (uint64_t)esp_timer_get_time());
    out[20] = (uint8_t)(int8_t)info->rx_ctrl.rssi;
    out[21] = (uint8_t)(int8_t)info->rx_ctrl.noise_floor;
    put_u16_le(&out[22], csi_len);
    put_u16_le(&out[24], subcarrier_count);
    put_u16_le(&out[26], 0);
    memcpy(&out[CSI_HEADER_LEN], info->buf, csi_len);
    return CSI_HEADER_LEN + csi_len;
}

static void csi_rx_callback(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    s_csi_callbacks++;
    int64_t now = esp_timer_get_time();
    int64_t min_interval_us = (int64_t)CONFIG_CSI_MIN_SEND_INTERVAL_MS * 1000;
    if ((now - s_last_send_us) < min_interval_us) {
        return;
    }
    s_last_send_us = now;

    if (s_udp_sock < 0) {
        return;
    }

    uint8_t packet[MAX_CSI_PACKET];
    size_t packet_len = build_csi_packet(info, packet, sizeof(packet));
    if (packet_len == 0) {
        return;
    }

    int sent = sendto(
        s_udp_sock,
        packet,
        packet_len,
        0,
        (struct sockaddr *)&s_udp_dest,
        sizeof(s_udp_dest));
    if (sent < 0) {
        s_csi_send_failures++;
        ESP_LOGW(TAG, "sendto failed: errno %d", errno);
    } else {
        s_csi_sent++;
    }
}

static void stats_task(void *arg)
{
    (void)arg;
    uint32_t last_callbacks = 0;
    uint32_t last_sent = 0;
    uint32_t last_failures = 0;

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        uint32_t callbacks = s_csi_callbacks;
        uint32_t sent = s_csi_sent;
        uint32_t failures = s_csi_send_failures;
        ESP_LOGI(
            TAG,
            "CSI callbacks/s=%lu sent/s=%lu total_sent=%lu send_failures/s=%lu",
            (unsigned long)(callbacks - last_callbacks),
            (unsigned long)(sent - last_sent),
            (unsigned long)sent,
            (unsigned long)(failures - last_failures));
        last_callbacks = callbacks;
        last_sent = sent;
        last_failures = failures;
    }
}

static void promiscuous_rx_callback(void *buf, wifi_promiscuous_pkt_type_t type)
{
    (void)buf;
    (void)type;
}

static void csi_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    wifi_promiscuous_filter_t filter = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA,
    };
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filter));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_rx_cb(promiscuous_rx_callback));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI enabled: node=%d min_interval=%d ms", CONFIG_CSI_NODE_ID, CONFIG_CSI_MIN_SEND_INTERVAL_MS);
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGE(TAG, "NVS init failed (%s); refusing to erase automatically", esp_err_to_name(ret));
        return;
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "Starting independent Wi-Fi CSI node");
    ESP_LOGI(TAG, "Node ID: %d", CONFIG_CSI_NODE_ID);
    wifi_init_sta();
    udp_init();
    csi_init();
    xTaskCreate(stats_task, "csi_stats", 3072, NULL, 5, NULL);
}
