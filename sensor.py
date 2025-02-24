import logging
import os
import csv
from datetime import datetime, timedelta
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from .const import *

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Meter Collector sensor from a config entry."""
    ip_address = config_entry.data["ip"]
    json_url = f"http://{ip_address}/{API_json}"
    image_url = f"http://{ip_address}/{API_img_alg}" 
    instance_name = config_entry.data["instance_name"]
    scan_interval = config_entry.options.get("scan_interval", DEFAULT_SCAN_INTERVAL)
    log_as_csv = config_entry.data.get("log_as_csv", True)
    save_images = config_entry.data.get("save_images", True)
    www_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../www/{DOMAIN}", instance_name))  # Save images in www folder

    # Create the www directory if it doesn't exist
    os.makedirs(www_dir, exist_ok=True)

    sensor = MeterCollectorSensor(
        hass=hass,
        ip_address=ip_address,
        json_url=json_url,
        image_url=image_url,
        www_dir=www_dir,
        scan_interval=scan_interval,
        instance_name=instance_name,
        log_as_csv=log_as_csv,
        save_images=save_images
    )
    async_add_entities([sensor])

    # Store the sensor in hass.data for service access
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN][instance_name] = sensor

class MeterCollectorSensor(Entity):
    """Representation of a Meter Collector sensor."""

    def __init__(self, hass, ip_address, json_url, image_url, www_dir, scan_interval, instance_name, log_as_csv, save_images):
        """Initialize the sensor."""
        self._hass = hass
        self._ip_address = ip_address
        self._json_url = json_url
        self._image_url = image_url
        self._www_dir = www_dir
        self._scan_interval = timedelta(seconds=scan_interval)
        self._instance_name = instance_name
        self.log_as_csv = log_as_csv
        self.save_images = save_images
        self._state = None
        self._attributes = {}
        self._last_update = None
        self._last_raw_value = None
        self._current_raw_value = None
        self._error_value = None
        self._latest_image_path = None

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

    @property
    def entity_picture(self):
        """Return the entity picture (local image path)."""
        return self._latest_image_path

    async def async_update(self):
        """Fetch new state data for the sensor."""
        try:
            # Throttle updates based on scan_interval
            if self._last_update and (datetime.now() - self._last_update) < self._scan_interval:
                _LOGGER.debug("Skipping update due to throttle")
                return

            session = async_get_clientsession(self._hass)

            # Fetch JSON data
            try:
                async with session.get(self._json_url) as response:
                    response.raise_for_status()
                    data = await response.json()
            except Exception as e:
                _LOGGER.error(f"Failed to fetch JSON data from {self._json_url}: {e}")
                self._state = "Error"
                self._attributes = {"error": f"Failed to fetch JSON data: {e}"}
                return

            # Validate JSON data
            if not data or not isinstance(data, dict):
                _LOGGER.error(f"Invalid JSON structure: Expected a dictionary, got {type(data)}")
                self._state = "Error"
                self._attributes = {"error": "Invalid JSON structure"}
                return

            # Get the first (and only) top-level key
            top_level_key = next(iter(data.keys()), None)
            if not top_level_key:
                _LOGGER.error("No top-level key found in JSON data")
                self._state = "Error"
                self._attributes = {"error": "No top-level key found in JSON data"}
                return

            # Extract values from the nested object
            nested_data = data.get(top_level_key, {})
            value = nested_data.get("value")
            raw_value = nested_data.get("raw")
            pre = nested_data.get("pre")
            error_value = nested_data.get("error", "no error")
            rate = nested_data.get("rate")
            timestamp = nested_data.get("timestamp")

            # Validate raw_value
            try:
                raw_value_float = float(raw_value)
            except (ValueError, TypeError) as e:
                _LOGGER.error(f"Invalid raw value received: {raw_value} ({e})")
                self._state = "Error"
                self._attributes = {"error": f"Invalid raw value: {raw_value}"}
                return

            # Skip if the new value is not greater than the last recorded value
            if self._last_raw_value is not None and raw_value_float <= self._last_raw_value:
                _LOGGER.debug(f"Skipping update: New value {raw_value} is not greater than last value {self._last_raw_value}")
                return

            # Get the current Unix epoch time
            unix_epoch = int(datetime.now().timestamp())

            # Save all values to CSV (Move to executor)
            if self.log_as_csv:
                csv_file = os.path.join(self._www_dir, "log.csv")
                try:
                    await self._hass.async_add_executor_job(
                        self._write_csv,
                        csv_file,
                        unix_epoch,
                        value,
                        raw_value,
                        pre,
                        error_value,
                        rate,
                        timestamp
                    )
                except Exception as e:
                    _LOGGER.error(f"Failed to write to CSV file {csv_file}: {e}")

            # Save image (Move to executor)
            if self.save_images:
                try:
                    # Fetch image from the remote URL
                    async with session.get(self._image_url) as image_response:
                        image_response.raise_for_status()
                        image_data = await image_response.read()

                    # Determine image file name
                    image_file = os.path.join(self._www_dir, f"{unix_epoch}_{raw_value}.jpg")
                    if error_value == "no error":
                        self._latest_image_path = f"/local/{DOMAIN}/{self._instance_name}/{unix_epoch}_{raw_value}.jpg"
                    else:
                        self._latest_image_path = f"/local/{DOMAIN}/{self._instance_name}/{unix_epoch}_{raw_value}_err.jpg"

                    # Save the image
                    await self._hass.async_add_executor_job(self._write_image, image_file, image_data)
                    await self._hass.async_add_executor_job(self._write_image, os.path.join(self._www_dir, "latest.jpg"), image_data)
                except Exception as e:
                    _LOGGER.error(f"Failed to fetch or save image: {e}")

            # Update state and attributes
            self._state = raw_value
            self._current_raw_value = raw_value_float

            # Include all relevant fields in attributes
            self._attributes = {
                "value": value,
                "raw": raw_value,
                "pre": pre,
                "error": error_value,
                "rate": rate,
                "timestamp": timestamp,
                "last_updated": datetime.now().isoformat(),
                "last_raw_value": self._last_raw_value,
                "current_raw_value": self._current_raw_value,
                "entity_picture": self._latest_image_path,
            }

            # Record the last update time and last raw value
            self._last_update = datetime.now()
            self._last_raw_value = raw_value_float

            # Set prevalue on error
            if error_value.lower() != "no error":
                _LOGGER.warning(f"Error detected: {error_value}")
                try:
                    prevalue_url = f"http://{self._ip_address}/setPreValue?numbers={self._instance_name}&value={pre}"
                    async with session.get(prevalue_url) as prevalue_response:
                        prevalue_response.raise_for_status()
                        data = await prevalue_response.text()
                except Exception as e:
                    _LOGGER.error(f"Failed to set prevalue: {e}")

        except Exception as e:
            _LOGGER.error(f"Unexpected error during update: {e}")
            self._state = "Error"
            self._attributes = {"error": str(e)}

    def _write_csv(self, csv_file, unix_epoch, value, raw_value, pre, error_value, rate, timestamp):
        """Helper method to write all values to a CSV file in an executor thread."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)

            # Check if the file exists
            file_exists = os.path.isfile(csv_file)

            # Open the file in append mode
            with open(csv_file, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write headers if the file doesn't exist
                if not file_exists:
                    csv_writer.writerow([
                        "Timestamp",
                        "Value",
                        "Raw Value",
                        "Pre",
                        "Error",
                        "Rate",
                        "Timestamp (JSON)"
                    ])

                # Write the row with all values
                csv_writer.writerow([
                    unix_epoch,
                    value,
                    raw_value,
                    pre,
                    error_value,
                    rate,
                    timestamp
                ])
        except Exception as e:
            _LOGGER.error(f"Failed to write to CSV file {csv_file}: {e}")

    def _write_image(self, image_file, image_data):
        """Helper method to write image data to a file in an executor thread."""
        try:
            with open(image_file, "wb") as imgfile:
                imgfile.write(image_data)
        except Exception as e:
            _LOGGER.error(f"Failed to write image to file {image_file}: {e}")