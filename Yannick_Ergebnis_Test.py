import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta, timezone

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhMDAwZmE2ZDJhZTQ0ZGNjYmRiYWU2ZGEzNTIyMjU5YSIsImlhdCI6MTcwNjI3MzEwOSwiZXhwIjoyMDIxNjMzMTA5fQ.AmCtbsBmqnYysnOTCc01_Ns_r6RSlV8c60GnPdaO8c8"
base_url = "172.30.232.3"
url = "http://" + base_url + ":8123/api/services/input_number/set_value"
time_shift = timedelta(hours=1)
headers = {
    "Authorization": f"Bearer {token}",
    "content-type": "application/json",
}


def data_query(entity_id):
    end_date = datetime.utcnow() + time_shift
    start_date = end_date - timedelta(weeks=2)

    params = {
        "filter_entity_id": entity_id,
        "start_time": start_date.isoformat(),
        "end_time": end_date.isoformat(),
    }

    response = requests.get("http://" + base_url + ":8123/api/history/period", headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()

        # Erstellen eines DataFrames aus den Daten
        records = []
        for entry in data:
            for item in entry:
                timestamp_str = item.get('last_changed')
                if timestamp_str:
                    # Konvertieren des Strings in ein datetime-Objekt
                    timestamp = datetime.fromisoformat(timestamp_str)
                    # Anpassen der Zeit
                    adjusted_timestamp = timestamp + timedelta(hours=1)
                    humidity_value = item.get('state')

                    # Check if humidity_value is a numerical and positive value
                    try:
                        humidity_value = float(humidity_value)
                        if humidity_value >= 0:
                            records.append({'Zeitpunkt': adjusted_timestamp, 'Bodenfeuchtigkeit (%)': humidity_value})
                    except ValueError:
                        # Ignore non-numerical values
                        pass

        df = pd.DataFrame(records)
        print(df)
        return df

    else:
        print(f"An error occured when querying: {response.status_code}, answer: {response.text}")
        return None


def regression(df, gg):
    ha_data = df
    # print(ha_data)

    # Umwandlung der Zeitstempel in das datetime-Format
    ha_data['Zeitpunkt'] = pd.to_datetime(ha_data['Zeitpunkt'])
    ha_data.sort_values(by='Zeitpunkt', inplace=True)

    # Berechnung der Änderung der Bodenfeuchtigkeit
    ha_data['Bodenfeuchtigkeitsänderung'] = ha_data['Bodenfeuchtigkeit (%)'].diff()

    # Setting date of last watering to the correct value, if none is found it's set to the first recorded date
    # code assumes that if the difference of the soil moisture is >= 10 the plant was watered
    try:
        date_last_watering = ha_data[ha_data['Bodenfeuchtigkeitsänderung'] > 10]['Zeitpunkt'].iloc[-1]
    except:
        date_last_watering = ha_data['Zeitpunkt'].iloc[0]
        print(f"kein Gießereignis gefunden. setze letzes_gießereignis auf ersten geloggten Tag: {date_last_watering}")

    date_last_watering = pd.Timestamp(date_last_watering)

    # Entfernen der Daten vor letztem Gießen
    new_data = ha_data[ha_data['Zeitpunkt'] > date_last_watering]

    # Zwischenkopie zur Festlegung auf die benötigten Daten
    data_copy = new_data[['Zeitpunkt', 'Bodenfeuchtigkeit (%)']].copy()

    # Umwandlung der Zeitstempel in numerische Werte (Unix-Zeit)
    data_copy['Zeitstempel'] = data_copy['Zeitpunkt'].astype(np.int64) // 10 ** 9

    # Eingabedaten für das Modell
    X = data_copy[['Zeitstempel']]
    y = data_copy['Bodenfeuchtigkeit (%)']

    # Modelltraining
    model = lr()
    model.fit(X, y)

    # Vorhersage für die nächsten Minuten
    last_recorded_timestamp = X['Zeitstempel'].iloc[-1]
    future_timestamps = np.arange(last_recorded_timestamp, last_recorded_timestamp + 180 * 60 * 8 * 10,
                                  60)  # 180 Minuten (3 Stunden) in die Zukunft
    future_timestamps_df = pd.DataFrame(future_timestamps, columns=['Zeitstempel'])
    predictions = model.predict(future_timestamps_df)

    # Berechnung der zukünftigen Zeitpunkte
    future_times = pd.to_datetime(future_timestamps, unit='s')
    future_prediction_series = pd.Series(predictions, index=future_times)

    # Plot der Daten
    """
    plt.figure(figsize=(20, 10))
    plt.plot(data_copy['Zeitpunkt'], y, label='Gemessene Daten', color='blue')
    plt.plot(future_prediction_series, label='Vorhersage', color='green')
    plt.title('Gemessene Bodenfeuchtigkeit und Vorhersage für die nächsten Stunden')
    plt.xlabel('Zeitpunkt')
    plt.ylabel('Bodenfeuchtigkeit (%)')
    plt.legend()
    plt.show()
    """

    # Bestimmung des ersten Zeitpunkts, an dem die Bodenfeuchtigkeit unter den Grenzwert fällt
    date_next_watering = future_prediction_series[future_prediction_series < gg].index[0] if not \
        future_prediction_series[future_prediction_series < gg].empty else None

    print(f"erster Zeitpunkt unterhalb der Grenze is: {date_next_watering}")
    if date_next_watering is None:
        date_next_watering = datetime.utcnow() + timedelta(days=10)  # 10 days in the future
        print("days until threshold was set to 10.")

    # return values
    values = {
        "date_next_watering": date_next_watering,
        "date_last_watering": date_last_watering
    }

    return values


def print_reg_result_to_HA(helper_id, reg_result):
    days_until_next_watering = (reg_result - datetime.utcnow() + time_shift).days
    if days_until_next_watering < 0:
        days_until_next_watering = 0

    param_write_reg_result = {
        "entity_id": helper_id,
        "value": days_until_next_watering,
    }

    writing_affirmation = requests.post(url, headers=headers, json=param_write_reg_result)
    if writing_affirmation.status_code == 200:
        print(f"{helper_id}: days until next watering successfully updated")
    else:
        print(f"{helper_id}: days until next watering could not be updated: ({writing_affirmation.status_code}")


def print_last_watering_to_HA(helper_id, last_watering):
    now_utc = datetime.now(timezone.utc)  # Make now_utc timezone-aware
    days_since_last_watering = (now_utc - last_watering).days
    print(f"days_since_last_watering: {days_since_last_watering}")
    if days_since_last_watering < 0:
        days_since_last_watering = 0

    param_write_last_watering = {
        "entity_id": helper_id,
        "value": days_since_last_watering,
    }

    writing_affirmation = requests.post(url, headers=headers, json=param_write_last_watering)
    if writing_affirmation.status_code == 200:
        print(f"{helper_id}: days since last watering successfully updated")
    else:
        print(f"{helper_id}: days since next watering could not be updated: ({writing_affirmation.status_code}")


def execute(plant_id, helper_id_next, helper_id_last, moisture_threshold):
    help_df = data_query(plant_id)

    # Check if the dataframe is None
    if help_df is not None:
        values = regression(help_df, moisture_threshold)

        date_next_watering = values["date_next_watering"]
        date_last_watering = values["date_last_watering"]

        print_reg_result_to_HA(helper_id_next, date_next_watering)
        print_last_watering_to_HA(helper_id_last, date_last_watering)
        print(f"Erfolg mit {plant_id}. Gießzeitpkt.: {date_next_watering}")
    else:
        print(f"Keine Daten für {plant_id}. Abbruch der weiteren Verarbeitung.")


execute("input_number.feuchtigkeitssensor", "input_number.giesstermin", "input_number.helper_last_watering",
        30)
