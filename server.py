import time
import requests
from mcp.server.fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("Demo 🚀")

def get_coordinates(location: str) -> tuple[float, float]:
    """Get latitude and longitude for a location using geocoding."""
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": location, "count": 1, "language": "en", "format": "json"}
        response = requests.get(geo_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            result = data["results"][0]
            return result["latitude"], result["longitude"]
        else:
            raise ValueError(f"Location '{location}' not found")
    except Exception as e:
        raise Exception(f"Geocoding error: {str(e)}")

@mcp.tool()
def get_weather(location: str) -> str:
    start = time.perf_counter()
    print(f"[MCP] get_weather called: {location}")

    try:
        lat, lon = get_coordinates(location)
        
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit"
        }
        response = requests.get(weather_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind_speed = current.get("wind_speed_10m", "N/A")
        
        # Simple weather code mapping
        weather_code = current.get("weather_code", 0)
        result = f"Weather in {location}: Code {weather_code}, {temp}°F, Humidity: {humidity}%, Wind: {wind_speed} mph"
        
    except Exception as e:
        result = f"Error fetching weather for {location}: {str(e)}"

    end = time.perf_counter()
    print(f"[MCP] tool latency: {(end - start)*1000:.1f} ms")

    return result

if __name__ == "__main__":
    # This runs the MCP server on port 8000 (default)
    mcp.run(transport="sse")