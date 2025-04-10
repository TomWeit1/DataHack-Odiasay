import glob
import pandas as pd
import numpy as np

def get_data(verbose: bool = False) -> pd.DataFrame:
    df_pv = get_pv_data(verbose)
    df_solar = get_solar_data(verbose)
    df_weather = get_weather_data(verbose)
    df = df_pv.join(df_solar, how='inner').join(df_weather, how='inner')
    df = (df - df.min()) / (df.max() - df.min())
    if verbose:
        print("\n## Complete Dataset:")
        print(df.info())
    return df


def get_weather_data(verbose: bool = False) -> pd.DataFrame:
    df_weather = pd.read_csv("data/world_weather_online.csv")
    df_weather['date_time'] = pd.to_datetime(df_weather['date_time'])
    df_weather.set_index('date_time', inplace=True)
    
    #for col in ['sunrise', 'sunset', 'moonrise', 'moonset']:  # note: might be 'moonset' instead of 'moonfall'
    #    df_weather[col] = pd.to_datetime(df_weather[col], format='%I:%M %p', errors='coerce')

# Calculate sun time in hours
    #df_weather['sun_time'] = (df_weather['sunset'] - df_weather['sunrise']).dt.total_seconds() / 3600

# Calculate moon time in hours, handling next-day moonset
    #df_weather['moon_time'] = (df_weather['moonset'] - df_weather['moonrise']).dt.total_seconds() / 3600
    #df_weather.loc[df_weather['moon_time'] < 0, 'moon_time'] += 24  # wrap around midnight

    df_weather.drop(columns=['moonrise', 'moonset', 'sunrise', 'sunset', 'location'], inplace =True)
    if verbose:
        print("\n## Weather data:")
        print(round(df_weather.describe(percentiles=[0.5]).transpose(), 2))
    return df_weather


def get_solar_data(verbose: bool = False) -> pd.DataFrame:
    df_solar = pd.concat([pd.read_csv(f) for f in glob.glob("data/nrel_solar_irradiance/*.csv")], ignore_index=True)
    df_solar['fractional_time'] = df_solar['Hour'] + df_solar['Minute'] / 60.0

# Create sine and cosine representations for capturing daily cycles
    df_solar['sin_time'] = np.sin(2 * np.pi * df_solar['fractional_time'] / 24)
    df_solar['cos_time'] = np.cos(2 * np.pi * df_solar['fractional_time'] / 24)
    df_solar['clear_sky_index_GHI'] = df_solar['GHI'] / df_solar['Clearsky GHI'].replace(0, np.nan)
    df_solar['clear_sky_index_GHI'].fillna(0, inplace=True)

# 4. Irradiance Loss Ratio:
# Proportion of lost irradiance relative to the clear-sky condition
    df_solar['irradiance_loss_ratio'] = (df_solar['Clearsky GHI'] - df_solar['GHI']) / df_solar['Clearsky GHI'].replace(0, np.nan)
    df_solar['irradiance_loss_ratio'].fillna(0, inplace=True)
    df_solar['date_time'] = pd.to_datetime(df_solar[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df_solar.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
    df_solar = df_solar.set_index('date_time').sort_index()
    if verbose:
        print("\n## Solar data:")
        print(round(df_solar.describe(percentiles=[0.5]).transpose(), 2))
    return df_solar


def get_pv_data(verbose: bool = False) -> pd.DataFrame:
    df_pv = pd.read_csv("data/PV_Output_Hannover_MA.csv")
    df_pv['date_time'] = pd.to_datetime(df_pv['Timestamp'], format="%b %d, %Y %I%p")
    df_pv.drop(columns=['Timestamp', 'City', 'County', 'State'], inplace=True)
    df_pv.rename(columns={'% Baseline': 'relative_power'}, inplace=True)
    df_pv.set_index('date_time', inplace=True)
    if verbose:
        print("\n## PV data:")
        print(df_pv.describe(percentiles=[0.5]).transpose())
    return df_pv
