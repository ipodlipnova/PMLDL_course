import pandas as pd

cities = pd.read_csv('/city.csv')[['city', 'region', 'geo_lat', 'geo_lon', 'population']]
cities = cities.replace(regex=r'\[.*\]', value='')
cities.loc[cities['city'].isnull(),'city'] = cities['region']

cities['population'] = pd.to_numeric(cities['population'])

res_cities = cities.sort_values('population', ascending=False).head(30)

coord = res_cities[['geo_lat', 'geo_lon']].to_numpy()
