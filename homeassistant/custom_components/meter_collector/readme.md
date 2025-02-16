# Meter Collector Integration for Home Assistant

This guide will walk you through the steps to install and configure the **Meter Collector** custom integration in Home Assistant.

## Installation

1. **Copy the Integration Folder:**
   - Download or clone the `meter_collector` custom component.
   - Copy the `meter_collector` folder into your Home Assistant `custom_components` directory.

2. **Set Permissions:**
   - Ensure the folder and files have the correct permissions (755).
     ```bash
     chmod -R 755 /homeassistant/custom_components/meter_collector
     ```

3. **Reboot Home Assistant:**
   - Restart Home Assistant (Hassio) to load the new integration.
     - You can do this via the **Settings > System > Restart** option in the Home Assistant UI.

4. **Add the Integration:**
   - Go to **Settings > Devices & Services > Integrations**.
   - Click **Add Integration** and search for **Meter Collector**.
   - Follow the prompts to configure the integration.

---

## Configuration

Once the integration is added, you will need to configure it with the following options:

### Required Configuration:
- **Instance**: A unique name for your sensor (e.g., `water_meter`).
- **Value URL**: The URL to fetch the meter value. Replace `{IP}` with the IP address of your meter device.
  ```
  http://{IP}/value?all=true&type=raw
  ```
- **Image URL**: The URL to fetch the meter image. Replace {IP} with the IP address of your meter device.
 ```
    http://{IP}/img_tmp/alg.jpg
 ```
Optional Configuration:
    Scan Interval: The time interval (in seconds) at which the integration will poll the meter for updates. Default is 3000 seconds (5min).
