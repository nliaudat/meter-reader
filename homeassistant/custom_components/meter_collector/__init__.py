from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from .sensor import MeterCollectorSensor

DOMAIN = "meter_collector"

async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Meter Collector integration."""
    # Register the service
    hass.services.async_register(
        DOMAIN,
        "collect_data",
        async_handle_collect_data,
        schema=vol.Schema({
            vol.Required("instance_name"): str,
        }),
    )
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Meter Collector from a config entry."""
    # Forward the setup to the sensor platform
    await hass.config_entries.async_forward_entry_setups(entry, ["sensor"])
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Forward the unload to the sensor platform
    await hass.config_entries.async_unload_platforms(entry, ["sensor"])
    return True

async def async_handle_collect_data(call: ServiceCall) -> None:
    """Handle the collect_data service call."""
    instance_name = call.data["instance_name"]
    # Find the sensor entity for the given instance
    sensor = next(
        (entity for entity in hass.data[DOMAIN].values()
        if isinstance(entity, MeterCollectorSensor) and entity.instance_name == instance_name
    ), None)
    if sensor:
        await sensor.async_update()
    else:
        _LOGGER.error(f"No sensor found for instance: {instance_name}")