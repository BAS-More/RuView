# RuView Sensing Platform — Final Shopping List

## Verified Prices (April 2026, AUD)

### Major Items

| # | Item | Qty | Store | Price/ea | Ship | Total | Link | Variant to Select |
|---|------|-----|-------|---------|------|-------|------|-------------------|
| 1 | SenseCAP Indicator D1Pro (hub) | 1 | Core Electronics AU | $149.95 | Free | $149.95 | https://core-electronics.com.au/sensecap-indicator-d1pro-iot-development-platform.html | — |
| 2 | LILYGO T-Deck Plus (nodes) | 3 | AliExpress LILYGO Official | $115.75 | $5.32 | $363.21 | https://www.aliexpress.com/item/1005007568074083.html | **915MHz Ulbox GNSS** |
| 3 | MR60BHA2 60GHz vital signs | 3 | Pakronics AU | $56.56 | AU | $169.68 | https://www.pakronics.com.au/products/114993387-mr60bha2-60ghz-mmwave-breathing-and-heartbeat-detection-sensor-kit-with-xiao-esp32c6-ss114993387 | — |

### Sensors (search Core Electronics AU first, AliExpress for cheaper)

| # | Item | Qty | Est Price/ea | Est Total | Search |
|---|------|-----|-------------|-----------|--------|
| 4 | BME680 breakout | 4 | ~$15 | ~$60 | Core Electronics: search "BME680" |
| 5 | LD2450 24GHz radar | 4 | ~$12 | ~$48 | Core Electronics: search "LD2450" or AliExpress |
| 6 | MLX90640 thermal array | 1 | ~$50 | ~$50 | Core Electronics: search "MLX90640" (110° wide-angle) |
| 7 | INMP441 I2S microphone | 4 | ~$3 | ~$12 | AliExpress: search "INMP441 I2S" |
| 8 | SCD40 CO2 sensor | 2 | ~$50 | ~$100 | Core Electronics: search "SCD40" (genuine Sensirion only!) |
| 9 | BH1750 light sensor | 4 | ~$2 | ~$8 | AliExpress: search "BH1750 GY-302" |
| 10 | Breadboard + jumper wires | 1 | ~$15 | ~$15 | eBay AU or AliExpress |

### Summary

| Category | Total AUD |
|----------|-----------|
| Major items (verified) | $682.84 |
| Sensors (estimated) | ~$293 |
| **Grand Total** | **~$976 AUD** |

## Purchase Order

### Order 1: AliExpress (LILYGO Official Store)
- 3x T-Deck Plus — select **915MHz Ulbox GNSS** variant
- Delivery: Apr 17-23 (fast!)
- Total: ~$363

### Order 2: Core Electronics AU
- 1x SenseCAP Indicator D1Pro
- 4x BME680
- 1x MLX90640
- 2x SCD40
- 4x LD2450 (if in stock, otherwise AliExpress)
- Total: ~$408

### Order 3: Pakronics AU (Melbourne)
- 3x MR60BHA2 60GHz sensor kit
- Total: ~$170

### Order 4: AliExpress (small sensors)
- 4x INMP441
- 4x BH1750
- 1x breadboard + jumper kit
- Total: ~$35

## WARNINGS
- T-Deck Plus: Select **915MHz** (not 868MHz which is EU)
- SCD40: Buy genuine Sensirion ONLY (many fakes on AliExpress)
- MLX90640: Get **110° wide-angle** version for room coverage
- MR60BHA2: Kit includes XIAO ESP32C6 (bonus MCU per sensor)
- T-Deck Plus: Will need touchscreen firmware fix (documented, 5 min)

## What This Gets You
- 4 sensing units (3 mobile + 1 hub)
- 107 monitoring capabilities
- WiFi CSI + 60GHz + 24GHz + thermal + acoustic + environmental
- LoRa mesh network between all nodes
- GPS on all mobile nodes
- Screens on all units (3x 2.8" + 1x 4")
- Batteries on mobile nodes
