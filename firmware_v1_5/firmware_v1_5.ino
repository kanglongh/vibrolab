/* vibrolab v1.5 · firmware_v1_5.ino — ESP32-S3 流式诊断.
 *
 * 协议 (stop-and-wait, 921600):
 *   PC -> 板: [0xA5][0x5A][2048 float32]  (8194 字节/窗)
 *   板 -> PC: "D <label> <conf> feat=<ms>ms infer=<us>us\n"   (英文标签, 给PC比对)
 *   OLED: 中文标签 (给人看) + 置信度 + 延迟
 *
 * OLED: SSD1306 128x64 I2C, SDA=4, SCL=5 (改 HAS_OLED=0 可关, 不需U8g2库) */
#define HAS_OLED 1  /* 1=启用 OLED 中文显示; 0=关 (省 U8g2 库依赖) */

#include <Arduino.h>
#include <string.h>
#include "model.h"
extern "C" {
#include "features.h"
#include "infer.h"
}

#if HAS_OLED
#include <U8g2lib.h>
#include <Wire.h>
#define SDA_PIN 4
#define SCL_PIN 5
static U8G2_SSD1306_128X64_NONAME_F_HW_I2C oled(U8G2_R0, U8X8_PIN_NONE, SCL_PIN, SDA_PIN);
static bool oled_ok = false;
/* 中文标签 (OLED 给人看). 串口仍用英文 VIBROLAB_LABEL_NAMES 给 PC 比对 */
static const char *LABEL_CN[10] = {
    "正常",   "滚珠007", "滚珠014", "滚珠021",
    "内圈007", "内圈014", "内圈021",
    "外圈007", "外圈014", "外圈021"
};
#endif

static float   window_buf[2048];
static unsigned long frame_count = 0;

static bool read_frame(float *win) {
    unsigned long t0 = millis();
    while (Serial.available() < 2) {
        if (millis() - t0 > 100) return false;
    }
    if (Serial.read() != 0xA5 || Serial.read() != 0x5A) return false;

    int got = 0;
    t0 = millis();
    while (got < 8192) {
        while (Serial.available() && got < 8192)
            ((char *)win)[got++] = Serial.read();
        if (millis() - t0 > 500) return false;
    }
    return true;
}

void setup() {
    Serial.setRxBufferSize(16384);
    Serial.begin(921600);
#if HAS_OLED
    oled.setBusClock(400000);   /* I2C 400kHz: sendBuffer 从~80ms 降到~20ms */
    oled.begin();
    oled_ok = true;   /* U8g2 begin 不可靠返回状态; 没 OLED 时 I2C 写静默失败, 不崩 */
    oled.setFont(u8g2_font_wqy15_t_gb2312);
    oled.clearBuffer();
    oled.drawUTF8(10, 22, "vibrolab v1.5");
    oled.drawUTF8(10, 44, "就绪");
    oled.sendBuffer();
    Serial.println("[OLED] init OK");
#endif
    delay(500);
    Serial.printf("READY v1.5  Bot-40+LR  %d类  fs=%d  window=%d\n",
                  VIBROLAB_N_CLASSES, VIBROLAB_FS, VIBROLAB_WINDOW);
}

void loop() {
    static bool first_frame_ok = false;
    if (!first_frame_ok) {
        static unsigned long last_ready = 0;
        if (millis() - last_ready > 2000) {
            Serial.printf("READY v1.5  Bot-40+LR  %d类  fs=%d  window=%d\n",
                          VIBROLAB_N_CLASSES, VIBROLAB_FS, VIBROLAB_WINDOW);
            last_ready = millis();
        }
    }

    if (!read_frame(window_buf)) { delay(1); return; }
    first_frame_ok = true;
    frame_count++;

    float feat[120];
    uint32_t t0 = micros();
    vbl_extract_cfd(window_buf, VIBROLAB_WINDOW, feat);
    uint32_t t1 = micros();
    float conf;
    int pred = vbl_infer(feat, &conf);
    uint32_t t2 = micros();

#if HAS_OLED
    if (oled_ok) {
        oled.clearBuffer();
        /* 顶: 实时波形 (2048 降采样到 128 点, 画在 y[2,22]) —— 故障冲击签名肉眼可见 */
        float wmin = 1e30f, wmax = -1e30f;
        for (int i = 0; i < 128; i++) {
            float v = window_buf[i * 16];
            if (v < wmin) wmin = v;
            if (v > wmax) wmax = v;
        }
        float wrange = (wmax - wmin) + 1e-3f;
        for (int i = 0; i < 127; i++) {
            int y0 = 22 - (int)((window_buf[i*16]     - wmin) / wrange * 20);
            int y1 = 22 - (int)((window_buf[(i+1)*16] - wmin) / wrange * 20);
            if (y0 < 2) y0 = 2;  if (y0 > 22) y0 = 22;
            if (y1 < 2) y1 = 2;  if (y1 > 22) y1 = 22;
            oled.drawLine(i, y0, i+1, y1);
        }
        oled.drawLine(0, 24, 127, 24);                 /* 分隔线 */
        /* 中: 中文标签 (大字, 居中) */
        oled.setFont(u8g2_font_wqy15_t_gb2312);
        int w = oled.getUTF8Width(LABEL_CN[pred]);
        oled.drawUTF8((128 - w) / 2, 42, LABEL_CN[pred]);
        /* 底: 置信度条 (y[46,52], 整宽) + 延迟/置信数值 */
        oled.drawFrame(0, 46, 128, 6);
        oled.drawBox(0, 46, (int)(conf * 128), 6);
        oled.setFont(u8g2_font_6x10_tr);
        char foot[32];
        snprintf(foot, sizeof(foot), "%.2f  %lums", conf, (unsigned long)(t2 - t1) / 1000);
        oled.drawUTF8(0, 63, foot);
        oled.sendBuffer();
    }
#endif

    Serial.printf("D %s %.3f feat=%lums infer=%luus\n",
                  VIBROLAB_LABEL_NAMES[pred], conf,
                  (unsigned long)(t1 - t0) / 1000, (unsigned long)(t2 - t1));
}
