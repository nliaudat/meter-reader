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
    ip_address = config_entry.data["ip"]
    value_url = f"http://{ip_address}/value?all=true&type=raw"
    image_url = f"http://{ip_address}/img_tmp/alg.jpg"
    error_url = f"http://{ip_address}/value?all=true&type=error"  # Add error URL
    instance_name = config_entry.data["instance_name"]
    scan_interval = config_entry.options.get("scan_interval", DEFAULT_SCAN_INTERVAL)
    data_dir = hass.config.path("custom_components/meter_collector/data", instance_name)

    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    sensor = MeterCollectorSensor(hass, value_url, image_url, error_url, data_dir, scan_interval, instance_name)
    async_add_entities([sensor])

    # Store the sensor in hass.data for service access
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN][instance_name] = sensor

class MeterCollectorSensor(Entity):
    """Representation of a Meter Collector sensor."""

    def __init__(self, hass, value_url, image_url, error_url, data_dir, scan_interval, instance_name):
        """Initialize the sensor."""
        self._hass = hass
        self._value_url = value_url
        self._image_url = image_url
        self._error_url = error_url 
        self._data_dir = data_dir
        self._scan_interval = timedelta(seconds=scan_interval)
        self._instance_name = instance_name
        self._state = None
        self._attributes = {}
        self._last_update = None
        self._last_raw_value = None
        self._current_raw_value = None
        self._error_value = None 

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
                    text_response = await response.text()
                    raw_value = text_response.split("\t")[-1].strip()

            try:
                raw_value_float = float(raw_value)
            except ValueError:
                _LOGGER.error(f"Invalid raw value received: {raw_value}")
                self._state = "Error"
                self._attributes = {"error": f"Invalid raw value: {raw_value}"}
                return

            # Fetch error value
            async with session.get(self._error_url) as error_response:
                error_response.raise_for_status()
                error_text = await error_response.text()
                self._error_value = error_text.split("\t")[-1].strip()

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

            # Save raw value to CSV (Move to executor)
            csv_file = os.path.join(self._data_dir, "log.csv")
            await self._hass.async_add_executor_job(self._write_csv, csv_file, unix_epoch, raw_value, self._error_value)

            # Save image (Move to executor)
            image_file = os.path.join(self._data_dir, f"{unix_epoch}_{raw_value}.jpg")
            await self._hass.async_add_executor_job(self._write_image, image_file, image_data)

            # Update state and attributes
            self._state = raw_value
            self._current_raw_value = raw_value_float
            self._attributes = {
                "image_url": self._image_url,
                "last_updated": datetime.now().isoformat(),
                "last_raw_value": self._last_raw_value,
                "current_raw_value": self._current_raw_value,
                "error_value": self._error_value  # Add error value to attributes
            }

            # Record the last update time and last raw value
            self._last_update = datetime.now()
            self._last_raw_value = raw_value_float

            # Log error if present
            if self._error_value.lower() != "no error":
                _LOGGER.warning(f"Error detected: {self._error_value}")

        except Exception as e:
            _LOGGER.error(f"Error fetching data: {e}")
            self._state = "Error"
            self._attributes = {"error": str(e)}

    def _write_csv(self, csv_file, unix_epoch, raw_value, error_value):
        """Helper method to write data to a CSV file in an executor thread."""
        with open(csv_file, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([unix_epoch, raw_value, error_value])  # Add error value to CSV

    def _write_image(self, image_file, image_data):
        """Helper method to write image data to a file in an executor thread."""
        with open(image_file, "wb") as imgfile:
            imgfile.write(image_data)