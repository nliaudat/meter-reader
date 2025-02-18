import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.config_entries import ConfigEntry
from .const import DOMAIN, DEFAULT_SCAN_INTERVAL

class MeterCollectorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Meter Collector."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            ip_address = user_input["ip"]
            
            # Validate IP address format
            if not self._is_valid_ip(ip_address):
                errors["ip"] = "invalid_ip"
            
            if not errors:
                # Construct URLs based on the provided IP
                user_input["value_url"] = f"http://{ip_address}/value?all=true&type=raw"
                user_input["image_url"] = f"http://{ip_address}/img_tmp/alg.jpg"
                user_input["error_url"] = f"http://{ip_address}/value?all=true&type=error"
                
                return self.async_create_entry(
                    title=user_input["instance_name"],  # Use instance name as the title
                    data=user_input
                )

        # Show the form
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("instance_name"): str, 
                vol.Required("ip"): str, 
                vol.Optional("scan_interval", default=DEFAULT_SCAN_INTERVAL): int,
            }),
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return MeterCollectorOptionsFlow(config_entry)

    def _is_valid_ip(self, ip):
        """Validate IP address format."""
        import re
        pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        return bool(re.match(pattern, ip))

class MeterCollectorOptionsFlow(config_entries.OptionsFlow):
    """Handle an options flow for Meter Collector."""

    def __init__(self, config_entry: ConfigEntry):
        """Initialize options flow."""
        self.config_entry = config_entry #depreciated

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
                vol.Required("ip", default=self.config_entry.data["ip"]): str,  # Include IP in the form
                vol.Optional("scan_interval", default=self.config_entry.options.get("scan_interval", DEFAULT_SCAN_INTERVAL)): int,
            })
        )