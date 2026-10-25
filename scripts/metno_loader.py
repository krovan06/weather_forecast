import requests


class MetNoLoader:

    BASE_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

    def __init__(self, user_agent: str):
        self.headers = {"User-Agent": user_agent}

    def fetch_forecast(self, lat: float, lon: float) -> dict:
        params = {
            "lat": round(lat, 4),
            "lon": round(lon, 4)
        }

        response = requests.get(
            self.BASE_URL,
            headers=self.headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()