import logging
import os
import csv
from datetime import datetime, timedelta
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from .const import DOMAIN, DEFAULT_SCAN_INTERVAL

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Meter Collector sensor from a config entry."""
    value_url = config_entry.data["value_url"]
    image_url = config_entry.data["image_url"]
    instance_name = config_entry.data["instance_name"]
    scan_interval = config_entry.options.get("scan_interval", DEFAULT_SCAN_INTERVAL)
    data_dir = hass.config.path("custom_components/meter_collector/data", instance_name)

    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    sensor = MeterCollectorSensor(hass, value_url, image_url, data_dir, scan_interval, instance_name)
    async_add_entities([sensor])

    # Store the sensor in hass.data for service access
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN][instance_name] = sensor

class MeterCollectorSensor(Entity):
    """Representation of a Meter Collector sensor."""

    def __init__(self, hass, value_url, image_url, data_dir, scan_interval, instance_name):
        """Initialize the sensor."""
        self._hass = hass
        self._value_url = value_url
        self._image_url = image_url
        self._data_dir = data_dir
        self._scan_interval = timedelta(seconds=scan_interval)
        self._instance_name = instance_name
        self._state = None
        self._attributes = {}
        self._last_update = None
        self._last_raw_value = None

    @property
    def name(self):
        """Return the name of the sensor."""
        return f"Meter Collector ({self._instance_name})"

    @property
    def state(self):
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self):
        """Return the state attributes."""
        return self._attributes

    async def async_update(self):
        """Fetch new state data for the sensor."""
        try:
            # Throttle updates based on scan_interval
            if self._last_update and (datetime.now() - self._last_update) < self._scan_interval:
                _LOGGER.debug("Skipping update due to throttle")
                return

            session = async_get_clientsession(self._hass)

            # Fetch raw value
            async with session.get(self._value_url) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()

                if "application/json" in content_type:
                    data = await response.json()
                    raw_value = data.get("rawValue")
                else:
                    # Assume plain text response
                    text_response = await response.text()
                    # Extract the numeric value after the tab character
                    raw_value = text_response.split("\t")[-1].strip()

            # Convert raw_value to float for comparison
            raw_value_float = float(raw_value)

            # Skip if the new value is not greater than the last recorded value
            if self._last_raw_value is not None and raw_value_float <= self._last_raw_value:
                _LOGGER.debug(f"Skipping update: New value {raw_value} is not greater than last value {self._last_raw_value}")
                return

            # Fetch image
            async with session.get(self._image_url) as image_response:
                image_response.raise_for_status()
                image_data = await image_response.read()

            # Get the current Unix epoch time
            unix_epoch = int(datetime.now().timestamp())

            # Save raw value to CSV
            csv_file = os.path.join(self._data_dir, "log.csv")
            with open(csv_file, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([unix_epoch, raw_value])

            # Save image with the name {unixepoch}_{value}.jpg
            image_file = os.path.join(self._data_dir, f"{unix_epoch}_{raw_value}.jpg")
            with open(image_file, "wb") as imgfile:
                imgfile.write(image_data)

            # Update state and attributes
            self._state = raw_value
            self._attributes = {
                "image_url": self._image_url,
                "last_updated": datetime.now().isoformat()
            }

            # Record the last update time and last raw value
            self._last_update = datetime.now()
            self._last_raw_value = raw_value_float

        except Exception as e:
            _LOGGER.error(f"Error fetching data: {e}")
            self._state = "Error"
            self._attributes = {"error": str(e)}