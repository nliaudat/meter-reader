import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.config_entries import ConfigEntry  # Import ConfigEntry
from .const import DOMAIN, DEFAULT_SCAN_INTERVAL

class MeterCollectorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Meter Collector."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            # Validate user input
            if not user_input["value_url"].startswith("http"):
                errors["value_url"] = "invalid_url"
            if not user_input["image_url"].startswith("http"):
                errors["image_url"] = "invalid_url"
            if not errors:
                # Input is valid, create the config entry
                return self.async_create_entry(
                    title=user_input["instance_name"],  # Use instance name as the title
                    data=user_input
                )

        # Show the form
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("instance_name"): str,  # Add instance name field
                vol.Required("value_url"): str,
                vol.Required("image_url"): str,
                vol.Optional("scan_interval", default=DEFAULT_SCAN_INTERVAL): int,
            }),
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return MeterCollectorOptionsFlow(config_entry)

class MeterCollectorOptionsFlow(config_entries.OptionsFlow):
    """Handle an options flow for Meter Collector."""

    def __init__(self, config_entry: ConfigEntry):
        """Initialize options flow."""
        # self.config_entry = config_entry #depreciated

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            # Update the config entry with new options
            return self.async_create_entry(title="", data=user_input)

        # Show the form with current values
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required("instance_name", default=self.config_entry.data["instance_name"]): str,
                vol.Required("value_url", default=self.config_entry.data["value_url"]): str,
                vol.Required("image_url", default=self.config_entry.data["image_url"]): str,
                vol.Optional("scan_interval", default=self.config_entry.options.get("scan_interval", DEFAULT_SCAN_INTERVAL)): int,
            })
        )