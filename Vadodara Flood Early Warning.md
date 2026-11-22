# Vadodara Flood Early Warning

Simulated water-level sensors at three flood-prone points. A trend model forecasts when each will cross its danger level, and alerts are written in English, Hindi and Gujarati.

Rainfall 0 mm/h

## Public alerts

No alerts. Raise the rainfall to see the system react.

Demo only: readings are simulated and the forecast is a linear fit over the last 8 readings (about 40 minutes). In a real deployment, ESP32 nodes with ultrasonic sensors would send readings over MQTT, a trained model (e.g. gradient boosting on level, rain and upstream data) would replace the trend fit, and an LLM would write the alert text. Street names and diversion routes here are illustrative.