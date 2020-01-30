### This analysis used Toronto's postal codes, as provided by Wikipedia, and Foursquare API to cluster Toronto's neighborhoods

```python
import pandas as pd
import numpy as np
```

## Segmenting and Clustering Neighborhoods in Toronto

### Part 1: Web Scraping


```python
from bs4 import BeautifulSoup
import requests
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
```


```python
# Reading in Data
df = pd.read_html(url)[0]
df = df.loc[(df.Borough != 'Not assigned')]
```


```python
# Altering Neighbourhood to equal Borough if not assigned
def myfunc(x,y):
    if x == 'Not assigned' and y != 'Not assigned':
        return y
    else:
        return x

df['Neighbourhood'] = df.apply(lambda x: myfunc(x.Neighbourhood, x.Borough), axis=1)    
```


```python
# Getting all like Neighbourhood and postal codes to align
unique = df['Postcode'].unique()
neighbourhood = np.zeros(len(unique))
borough = np.zeros(len(unique))   
```


```python
df2 = pd.DataFrame(df.groupby(['Postcode','Borough'])['Neighbourhood'].agg(', '.join)).reset_index()
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.shape
```




    (103, 3)



## Part 2: Locations


```python
# Reading in provided csv
locations = pd.read_csv("Geospatial_Coordinates.csv")

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>




```python
import geocoder
```


```python
g = geocoder.arcgis('Brookwood, Atlanta, GA')
```


```python
g.geojson
```




    {'type': 'FeatureCollection',
     'features': [{'type': 'Feature',
       'properties': {'address': 'Brookwood, Atlanta, Georgia',
        'bbox': [-84.40382999999999,
         33.79247000000003,
         -84.38382999999997,
         33.812470000000026],
        'confidence': 7,
        'lat': 33.80247000000003,
        'lng': -84.39382999999998,
        'ok': True,
        'quality': 'Locality',
        'raw': {'name': 'Brookwood, Atlanta, Georgia',
         'extent': {'xmin': -84.40382999999999,
          'ymin': 33.79247000000003,
          'xmax': -84.38382999999997,
          'ymax': 33.812470000000026},
         'feature': {'geometry': {'x': -84.39382999999998, 'y': 33.80247000000003},
          'attributes': {'Score': 100, 'Addr_Type': 'Locality'}}},
        'score': 100,
        'status': 'OK'},
       'bbox': [-84.40382999999999,
        33.79247000000003,
        -84.38382999999997,
        33.812470000000026],
       'geometry': {'type': 'Point',
        'coordinates': [-84.39382999999998, 33.80247000000003]}}]}




```python
df3 = df2.join(locations, how = "left", lsuffix= 'Postcode', rsuffix='Postal Code').drop('Postal Code', axis = 1)
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



## Part 3: Mapping


```python
import folium
from geopy.geocoders import Nominatim
```

### Subsetting to just the Toronto Boroughs


```python
# Subsetting to just the Toronto Boroughs
df4 = df3[df3['Borough'].str.contains("Toronto")].drop(70)
```

### Getting Traverse City Location


```python
address = 'Traverse City, MI'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Toronto are 43.653963, -79.387207.
    

### Starting the search via Foursquare API


```python

```

### Defining Function to get values for multiple boroughs


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```

### Searching for the Toronto Data from Foursquare


```python
# type your answer here
LIMIT = 100
toronto_venues = getNearbyVenues(names=df4['Neighbourhood'],
                                   latitudes=df4['Latitude'],
                                   longitudes=df4['Longitude']
                                  )
```

    The Beaches
    The Danforth West, Riverdale
    The Beaches West, India Bazaar
    Studio District
    Lawrence Park
    Davisville North
    North Toronto West
    Davisville
    Moore Park, Summerhill East
    Deer Park, Forest Hill SE, Rathnelly, South Hill, Summerhill West
    Rosedale
    Cabbagetown, St. James Town
    Church and Wellesley
    Harbourfront
    Ryerson, Garden District
    St. James Town
    Berczy Park
    Central Bay Street
    Adelaide, King, Richmond
    Harbourfront East, Toronto Islands, Union Station
    Design Exchange, Toronto Dominion Centre
    Commerce Court, Victoria Hotel
    Roselawn
    Forest Hill North, Forest Hill West
    The Annex, North Midtown, Yorkville
    Harbord, University of Toronto
    Chinatown, Grange Park, Kensington Market
    CN Tower, Bathurst Quay, Island airport, Harbourfront West, King and Spadina, Railway Lands, South Niagara
    Stn A PO Boxes 25 The Esplanade
    Christie
    Dovercourt Village, Dufferin
    Little Portugal, Trinity
    Brockton, Exhibition Place, Parkdale Village
    High Park, The Junction South
    Parkdale, Roncesvalles
    Runnymede, Swansea
    Queen's Park
    Business Reply Mail Processing Centre 969 Eastern
    

The returned some Venue Categories of **Venue Categeory**, so I am just removing for now. Out of all entries, it was only n = 4.


```python
# Dropping the Venue Category 'Neighborhood'
toronto_venues = toronto_venues[toronto_venues['Venue Category'] != 'Neighborhood']
```

### Compling the One Hot Encoding for all venues


```python
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
#list(toronto_onehot.columns)

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>...</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The Danforth West, Riverdale</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 232 columns</p>
</div>



### Grouping for the most common venues


```python
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
```


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelaide, King, Richmond</td>
      <td>Coffee Shop</td>
      <td>Bar</td>
      <td>Café</td>
      <td>Steakhouse</td>
      <td>Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Hotel</td>
      <td>Thai Restaurant</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Cocktail Bar</td>
      <td>Cheese Shop</td>
      <td>Bakery</td>
      <td>Beer Bar</td>
      <td>Farmers Market</td>
      <td>Seafood Restaurant</td>
      <td>Steakhouse</td>
      <td>Café</td>
      <td>Butcher</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Grocery Store</td>
      <td>Stadium</td>
      <td>Burrito Place</td>
      <td>Restaurant</td>
      <td>Climbing Gym</td>
      <td>Performing Arts Venue</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>Yoga Studio</td>
      <td>Auto Workshop</td>
      <td>Park</td>
      <td>Comic Shop</td>
      <td>Pizza Place</td>
      <td>Recording Studio</td>
      <td>Restaurant</td>
      <td>Burrito Place</td>
      <td>Brewery</td>
      <td>Light Rail Station</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>Airport Lounge</td>
      <td>Airport Service</td>
      <td>Airport Terminal</td>
      <td>Boutique</td>
      <td>Boat or Ferry</td>
      <td>Bar</td>
      <td>Rental Car Location</td>
      <td>Plane</td>
      <td>Coffee Shop</td>
      <td>Harbor / Marina</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.cluster import KMeans
```

### Doing K Means Clustering Now


```python
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4,
           0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0])



### Adding clustering labels onto the results from K-Means


```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df4

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on = 'Neighbourhood')

toronto_merged.head() # check the last columns!
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>1</td>
      <td>Park</td>
      <td>Trail</td>
      <td>Health Food Store</td>
      <td>Pub</td>
      <td>Department Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>41</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Furniture / Home Store</td>
      <td>Yoga Studio</td>
      <td>Fruit &amp; Vegetable Store</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>42</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
      <td>0</td>
      <td>Park</td>
      <td>Board Shop</td>
      <td>Steakhouse</td>
      <td>Sushi Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Brewery</td>
      <td>Pub</td>
      <td>Liquor Store</td>
      <td>Fast Food Restaurant</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>43</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
      <td>0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Gastropub</td>
      <td>Brewery</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>American Restaurant</td>
      <td>Park</td>
      <td>Seafood Restaurant</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>44</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
      <td>4</td>
      <td>Park</td>
      <td>Swim School</td>
      <td>Bus Line</td>
      <td>Yoga Studio</td>
      <td>Dim Sum Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### Vizualizing the output on a map


```python
import matplotlib.cm as cm
import matplotlib.colors as colors
```


```python
# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_toronto)
       
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZiA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZicsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzQ0OTdmZTVjNWFjYzQ5OWM4ZjA1NjA2MDllMWFlMDEzID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hOGVhMDA0MDUwYzg0YTM3OWQ0NzYzMjM5YWMzMmM5ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3NjM1NzM5OTk5OTk5LC03OS4yOTMwMzEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MwNTYzY2ZkZGIzMDQyN2Y4OWVjOWI0NGYxNzljYmJmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE0NDczYzE3YThkMzQ0YzRiNzdmMGZkNWExNWM4YTE3ID0gJCgnPGRpdiBpZD0iaHRtbF8xNDQ3M2MxN2E4ZDM0NGM0Yjc3ZjBmZDVhMTVjOGExNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEJlYWNoZXMgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMDU2M2NmZGRiMzA0MjdmODllYzliNDRmMTc5Y2JiZi5zZXRDb250ZW50KGh0bWxfMTQ0NzNjMTdhOGQzNDRjNGI3N2YwZmQ1YTE1YzhhMTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYThlYTAwNDA1MGM4NGEzNzlkNDc2MzIzOWFjMzJjOWQuYmluZFBvcHVwKHBvcHVwX2MwNTYzY2ZkZGIzMDQyN2Y4OWVjOWI0NGYxNzljYmJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NmN2U0M2YwMGU4MTRmNGU4NDA1OTdiYzBkNzY3OGIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTU3MSwtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFiNjQ4ZGM3ZGJlMDRhNTlhYTg2YTE3ZWFmNjc3MzAxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FlZTg4NjcyZjcwMjQ1NDQ4Yjg4NTdhMGIwMWI3NjMyID0gJCgnPGRpdiBpZD0iaHRtbF9hZWU4ODY3MmY3MDI0NTQ0OGI4ODU3YTBiMDFiNzYzMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIERhbmZvcnRoIFdlc3QsIFJpdmVyZGFsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFiNjQ4ZGM3ZGJlMDRhNTlhYTg2YTE3ZWFmNjc3MzAxLnNldENvbnRlbnQoaHRtbF9hZWU4ODY3MmY3MDI0NTQ0OGI4ODU3YTBiMDFiNzYzMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZjdlNDNmMDBlODE0ZjRlODQwNTk3YmMwZDc2NzhiMi5iaW5kUG9wdXAocG9wdXBfMWI2NDhkYzdkYmUwNGE1OWFhODZhMTdlYWY2NzczMDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWI3ZjQxYzZjZWM3NDgxOWI0ZTRiZjRjMGVlZGE1OWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMTdhYTVhZGQ2YWQ0MTRiYWM3ZGM3YWFiOTk3YjcxOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNmZiMzVmYTY3MGQ0ODI4YTA0MmUyM2ViNTM3YjkyOSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTZmYjM1ZmE2NzBkNDgyOGEwNDJlMjNlYjUzN2I5MjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzIFdlc3QsIEluZGlhIEJhemFhciBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IxN2FhNWFkZDZhZDQxNGJhYzdkYzdhYWI5OTdiNzE5LnNldENvbnRlbnQoaHRtbF8xNmZiMzVmYTY3MGQ0ODI4YTA0MmUyM2ViNTM3YjkyOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYjdmNDFjNmNlYzc0ODE5YjRlNGJmNGMwZWVkYTU5Yy5iaW5kUG9wdXAocG9wdXBfYjE3YWE1YWRkNmFkNDE0YmFjN2RjN2FhYjk5N2I3MTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjk0NjljZTJkMTgzNDE3NWJjMDc3NTc3MmY4MzIxOTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDYyNjE4ZjAzMzk4NGU1NmI2ZTQyOTI0Njk3OTgxOWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzcxZjVlZjY4NDBlNGYxYzg2OTgwMGQ2MDI0OTVmNGYgPSAkKCc8ZGl2IGlkPSJodG1sXzc3MWY1ZWY2ODQwZTRmMWM4Njk4MDBkNjAyNDk1ZjRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNjI2MThmMDMzOTg0ZTU2YjZlNDI5MjQ2OTc5ODE5Yi5zZXRDb250ZW50KGh0bWxfNzcxZjVlZjY4NDBlNGYxYzg2OTgwMGQ2MDI0OTVmNGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjk0NjljZTJkMTgzNDE3NWJjMDc3NTc3MmY4MzIxOTAuYmluZFBvcHVwKHBvcHVwXzA2MjYxOGYwMzM5ODRlNTZiNmU0MjkyNDY5Nzk4MTliKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M5NzRiZWQxMGM5MTRhYjE4NmRjMzJlOGFjNjM2M2RhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MDViNmFkOTk4Y2M0MjE5OGVmYWUwYjdlOTc1YjQwYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMTUyYzMxNjBmYTA0ZTg4OGRmYjM3ZWNmYmRkMGZkZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMzE1MmMzMTYwZmEwNGU4ODhkZmIzN2VjZmJkZDBmZGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIFBhcmsgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MDViNmFkOTk4Y2M0MjE5OGVmYWUwYjdlOTc1YjQwYS5zZXRDb250ZW50KGh0bWxfMzE1MmMzMTYwZmEwNGU4ODhkZmIzN2VjZmJkZDBmZGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzk3NGJlZDEwYzkxNGFiMTg2ZGMzMmU4YWM2MzYzZGEuYmluZFBvcHVwKHBvcHVwXzUwNWI2YWQ5OThjYzQyMTk4ZWZhZTBiN2U5NzViNDBhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc3NWZhZDBhNzRlMjQ0YjE4MDUzNzI1MmQzYWM5Zjc3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMWMyNzc0MGY2Y2E0MjYxYTljNzEzMWM5MDI3YTEzYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZmFiM2ZlYTU3MjQ0OTQ1YWM0MzRhNDFmYWM3MTllNiA9ICQoJzxkaXYgaWQ9Imh0bWxfZmZhYjNmZWE1NzI0NDk0NWFjNDM0YTQxZmFjNzE5ZTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGggQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMWMyNzc0MGY2Y2E0MjYxYTljNzEzMWM5MDI3YTEzYS5zZXRDb250ZW50KGh0bWxfZmZhYjNmZWE1NzI0NDk0NWFjNDM0YTQxZmFjNzE5ZTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzc1ZmFkMGE3NGUyNDRiMTgwNTM3MjUyZDNhYzlmNzcuYmluZFBvcHVwKHBvcHVwXzMxYzI3NzQwZjZjYTQyNjFhOWM3MTMxYzkwMjdhMTNhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU3ZWNjMGU4NzdhNTQxNGE5MmEyNmJkNjU5MWU4MDI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjIwZWM2OGFmMjljNDNmYjgwZjYxNWMyMzcxYWNiZWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDY2M2ZiYjViZDNkNGUzZjlmNjNiMDk2ZDkxMTYzZjAgPSAkKCc8ZGl2IGlkPSJodG1sXzQ2NjNmYmI1YmQzZDRlM2Y5ZjYzYjA5NmQ5MTE2M2YwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMjBlYzY4YWYyOWM0M2ZiODBmNjE1YzIzNzFhY2JlYS5zZXRDb250ZW50KGh0bWxfNDY2M2ZiYjViZDNkNGUzZjlmNjNiMDk2ZDkxMTYzZjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTdlY2MwZTg3N2E1NDE0YTkyYTI2YmQ2NTkxZTgwMjguYmluZFBvcHVwKHBvcHVwX2IyMGVjNjhhZjI5YzQzZmI4MGY2MTVjMjM3MWFjYmVhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNjNTIzNWU3YzRhMDQzYjZiYjYxNDQ2ZGQyYjEwZWE4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA0MzI0NCwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MTNkZmQ5YTdiZTA0ZjVhODQ3MTFlNjBlNTM2MGNlZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMDBlZGE1MmJhZDg0MGZlOWEzZWI4NzFhZjc3M2M2YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzAwZWRhNTJiYWQ4NDBmZTlhM2ViODcxYWY3NzNjNmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MTNkZmQ5YTdiZTA0ZjVhODQ3MTFlNjBlNTM2MGNlZS5zZXRDb250ZW50KGh0bWxfMzAwZWRhNTJiYWQ4NDBmZTlhM2ViODcxYWY3NzNjNmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2M1MjM1ZTdjNGEwNDNiNmJiNjE0NDZkZDJiMTBlYTguYmluZFBvcHVwKHBvcHVwXzYxM2RmZDlhN2JlMDRmNWE4NDcxMWU2MGU1MzYwY2VlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkyNGY0YmVkOGJmYTQ4NjY4MzQyZjRlMWE4YjA1OThlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmVhZWYyZjgzMmRiNDI2NTk2ZjhjMmIzNjQ2ZWNlMjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmE2ZjdmYzRlM2MxNGQ0ZTk2MmE1NWI0ZGVmMmNkMDQgPSAkKCc8ZGl2IGlkPSJodG1sXzJhNmY3ZmM0ZTNjMTRkNGU5NjJhNTViNGRlZjJjZDA0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb29yZSBQYXJrLCBTdW1tZXJoaWxsIEVhc3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZWFlZjJmODMyZGI0MjY1OTZmOGMyYjM2NDZlY2UyNi5zZXRDb250ZW50KGh0bWxfMmE2ZjdmYzRlM2MxNGQ0ZTk2MmE1NWI0ZGVmMmNkMDQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTI0ZjRiZWQ4YmZhNDg2NjgzNDJmNGUxYThiMDU5OGUuYmluZFBvcHVwKHBvcHVwXzJlYWVmMmY4MzJkYjQyNjU5NmY4YzJiMzY0NmVjZTI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkwYTQzOGM5YzI1NjQ4OGNiOTA4MDU2NGE5NjAzY2M2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmZkZjEzMWIyNmYwNDBjZjgyZGI5NzM3NjI5ZjhkYzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTE1OWE2Yjc4MjdhNDUzNmI5YTZmOGRlMzg4YmYzYzIgPSAkKCc8ZGl2IGlkPSJodG1sX2ExNTlhNmI3ODI3YTQ1MzZiOWE2ZjhkZTM4OGJmM2MyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWVyIFBhcmssIEZvcmVzdCBIaWxsIFNFLCBSYXRobmVsbHksIFNvdXRoIEhpbGwsIFN1bW1lcmhpbGwgV2VzdCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZmZGYxMzFiMjZmMDQwY2Y4MmRiOTczNzYyOWY4ZGMyLnNldENvbnRlbnQoaHRtbF9hMTU5YTZiNzgyN2E0NTM2YjlhNmY4ZGUzODhiZjNjMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MGE0MzhjOWMyNTY0ODhjYjkwODA1NjRhOTYwM2NjNi5iaW5kUG9wdXAocG9wdXBfNmZkZjEzMWIyNmYwNDBjZjgyZGI5NzM3NjI5ZjhkYzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2I2NDRkMjc0ZDEyNDg0ZWFiNTAwNWNjZDFkM2RhODcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYWIyZjYxNmI5NDY0Mzg0YmY1YWI0MDJlODBlZGVlOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YjA0MDZkZTJlNWU0M2FlOTU1MjM3OWU0NTViNDg4YSA9ICQoJzxkaXYgaWQ9Imh0bWxfOGIwNDA2ZGUyZTVlNDNhZTk1NTIzNzllNDU1YjQ4OGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VkYWxlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWFiMmY2MTZiOTQ2NDM4NGJmNWFiNDAyZTgwZWRlZTguc2V0Q29udGVudChodG1sXzhiMDQwNmRlMmU1ZTQzYWU5NTUyMzc5ZTQ1NWI0ODhhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NiNjQ0ZDI3NGQxMjQ4NGVhYjUwMDVjY2QxZDNkYTg3LmJpbmRQb3B1cChwb3B1cF8xYWIyZjYxNmI5NDY0Mzg0YmY1YWI0MDJlODBlZGVlOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83YTQyOTdmYWE1ZDM0OTgzYmIzMDFiMWE5MTkwZGVhZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZmQyZjgxOTkyMzY0MGQxYTVkZmM2NjM1YWRkYjdmNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MjgxNWRjMTc1ZDU0MjkwYjY4NzIzYjJjNWRhYWU2MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzI4MTVkYzE3NWQ1NDI5MGI2ODcyM2IyYzVkYWFlNjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhYmJhZ2V0b3duLCBTdC4gSmFtZXMgVG93biBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVmZDJmODE5OTIzNjQwZDFhNWRmYzY2MzVhZGRiN2Y0LnNldENvbnRlbnQoaHRtbF83MjgxNWRjMTc1ZDU0MjkwYjY4NzIzYjJjNWRhYWU2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83YTQyOTdmYWE1ZDM0OTgzYmIzMDFiMWE5MTkwZGVhZS5iaW5kUG9wdXAocG9wdXBfNWZkMmY4MTk5MjM2NDBkMWE1ZGZjNjYzNWFkZGI3ZjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjY5ZDY5NzMxNGU3NGNkMWEzNDRlNjA0MGNiZjkxZjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZWU3OTlmZGVlMDk0ODNhYjI2ZjJjY2U4MDg2YTBlYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80Y2QyODRjZTgzYzI0Mzg4OGJhZmQxYTg2YTVjYjNmMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNGNkMjg0Y2U4M2MyNDM4ODhiYWZkMWE4NmE1Y2IzZjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGVlNzk5ZmRlZTA5NDgzYWIyNmYyY2NlODA4NmEwZWEuc2V0Q29udGVudChodG1sXzRjZDI4NGNlODNjMjQzODg4YmFmZDFhODZhNWNiM2YzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y2OWQ2OTczMTRlNzRjZDFhMzQ0ZTYwNDBjYmY5MWY5LmJpbmRQb3B1cChwb3B1cF80ZWU3OTlmZGVlMDk0ODNhYjI2ZjJjY2U4MDg2YTBlYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNmFkNzVkZjVkNGE0N2I2YTU5YzBhNTE2MTYxODEyYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzY2OTUzNDY1YzhkNDdmM2FlNWVhYTAwNjEyMjI2YTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWJmOGNlNjBhMzA2NGVlNDlhY2UwNTQ5MTU1M2EzZGUgPSAkKCc8ZGl2IGlkPSJodG1sX2FiZjhjZTYwYTMwNjRlZTQ5YWNlMDU0OTE1NTNhM2RlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NjY5NTM0NjVjOGQ0N2YzYWU1ZWFhMDA2MTIyMjZhOC5zZXRDb250ZW50KGh0bWxfYWJmOGNlNjBhMzA2NGVlNDlhY2UwNTQ5MTU1M2EzZGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDZhZDc1ZGY1ZDRhNDdiNmE1OWMwYTUxNjE2MTgxMmIuYmluZFBvcHVwKHBvcHVwXzc2Njk1MzQ2NWM4ZDQ3ZjNhZTVlYWEwMDYxMjIyNmE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU2MTg4ZWMwZGI3NTQ2ZTQ5ZWVmOTZiMDZiYmZlZjQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGIwZWE2MDc2M2IxNGIxZWI2YzJhYjhiMGQxMDU1NmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzU3YTExMjljOGVmNGFjMDk0YTc1NmQxMzExYTU0NGQgPSAkKCc8ZGl2IGlkPSJodG1sXzM1N2ExMTI5YzhlZjRhYzA5NGE3NTZkMTMxMWE1NDRkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SeWVyc29uLCBHYXJkZW4gRGlzdHJpY3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YjBlYTYwNzYzYjE0YjFlYjZjMmFiOGIwZDEwNTU2ZC5zZXRDb250ZW50KGh0bWxfMzU3YTExMjljOGVmNGFjMDk0YTc1NmQxMzExYTU0NGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTYxODhlYzBkYjc1NDZlNDllZWY5NmIwNmJiZmVmNDYuYmluZFBvcHVwKHBvcHVwXzRiMGVhNjA3NjNiMTRiMWViNmMyYWI4YjBkMTA1NTZkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJlYjliOGE0YWFiMzQzNDk4ZGY1NGU0NjEzZDYzZDgxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84M2RjZGY5ZDM0MTc0YTY0OTNkYWFjZDczYWZlYWQ3NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNzFkNTVkYzM2NmY0ZTlkOTA5ZWI0ZTgzMWRhY2EzZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzcxZDU1ZGMzNjZmNGU5ZDkwOWViNGU4MzFkYWNhM2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODNkY2RmOWQzNDE3NGE2NDkzZGFhY2Q3M2FmZWFkNzYuc2V0Q29udGVudChodG1sXzM3MWQ1NWRjMzY2ZjRlOWQ5MDllYjRlODMxZGFjYTNkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJlYjliOGE0YWFiMzQzNDk4ZGY1NGU0NjEzZDYzZDgxLmJpbmRQb3B1cChwb3B1cF84M2RjZGY5ZDM0MTc0YTY0OTNkYWFjZDczYWZlYWQ3Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iYjI0MThlODUxNmU0NWYxYTU3NDU2MTFiOTU4YzRmZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MDIzODMxY2E3NDQ0ZWQzOGMxMzIwYjAzMDdjMDE5NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNTc2ZjdmNDEzMWI0ZmRlOTljOTM2M2ZlZmY3MjVmZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMTU3NmY3ZjQxMzFiNGZkZTk5YzkzNjNmZWZmNzI1ZmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcmN6eSBQYXJrIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDAyMzgzMWNhNzQ0NGVkMzhjMTMyMGIwMzA3YzAxOTUuc2V0Q29udGVudChodG1sXzE1NzZmN2Y0MTMxYjRmZGU5OWM5MzYzZmVmZjcyNWZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JiMjQxOGU4NTE2ZTQ1ZjFhNTc0NTYxMWI5NThjNGZkLmJpbmRQb3B1cChwb3B1cF80MDIzODMxY2E3NDQ0ZWQzOGMxMzIwYjAzMDdjMDE5NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MjMyZDExMDRiZWE0M2U5OWVlNDJjMDU3NjA2ZWZjMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjc3Yzk3OGQxNjQwNGJjYTg1ZWVhN2MyY2E3ZTE2ZDUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjBhMjE2OTk4ZjZkNDNiYzgyMjFmZDQ1NjhhMTA3MTAgPSAkKCc8ZGl2IGlkPSJodG1sXzIwYTIxNjk5OGY2ZDQzYmM4MjIxZmQ0NTY4YTEwNzEwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNzdjOTc4ZDE2NDA0YmNhODVlZWE3YzJjYTdlMTZkNS5zZXRDb250ZW50KGh0bWxfMjBhMjE2OTk4ZjZkNDNiYzgyMjFmZDQ1NjhhMTA3MTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTIzMmQxMTA0YmVhNDNlOTllZTQyYzA1NzYwNmVmYzIuYmluZFBvcHVwKHBvcHVwX2Y3N2M5NzhkMTY0MDRiY2E4NWVlYTdjMmNhN2UxNmQ1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEwMzNmOWQxN2Q4MTQ1NGRiMmY5MTc2NDUyMjM2MmQ5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjM5MzA4M2YyZThiNDY2OTg0Y2JlMGFjMGExYjZhZTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmRhMmM3ZDIyMjEwNGIwYWIyZTUzMzhiNmIzNzJmMWQgPSAkKCc8ZGl2IGlkPSJodG1sX2JkYTJjN2QyMjIxMDRiMGFiMmU1MzM4YjZiMzcyZjFkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZGVsYWlkZSwgS2luZywgUmljaG1vbmQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMzkzMDgzZjJlOGI0NjY5ODRjYmUwYWMwYTFiNmFlMy5zZXRDb250ZW50KGh0bWxfYmRhMmM3ZDIyMjEwNGIwYWIyZTUzMzhiNmIzNzJmMWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTAzM2Y5ZDE3ZDgxNDU0ZGIyZjkxNzY0NTIyMzYyZDkuYmluZFBvcHVwKHBvcHVwXzIzOTMwODNmMmU4YjQ2Njk4NGNiZTBhYzBhMWI2YWUzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBkYzQwMjc4ZDdiZTRkZmI5NDFiNzExMWFlOWMzOTA0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzJiYzEwZTgwM2MwNDE3MGI5ZmNjM2ZhOTUzOTdkMTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTBjN2Y3ZDZkNjE4NGI2YzgxODMwOTQ2MDJkZDQyNTggPSAkKCc8ZGl2IGlkPSJodG1sXzEwYzdmN2Q2ZDYxODRiNmM4MTgzMDk0NjAyZGQ0MjU4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgRWFzdCwgVG9yb250byBJc2xhbmRzLCBVbmlvbiBTdGF0aW9uIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzJiYzEwZTgwM2MwNDE3MGI5ZmNjM2ZhOTUzOTdkMTUuc2V0Q29udGVudChodG1sXzEwYzdmN2Q2ZDYxODRiNmM4MTgzMDk0NjAyZGQ0MjU4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBkYzQwMjc4ZDdiZTRkZmI5NDFiNzExMWFlOWMzOTA0LmJpbmRQb3B1cChwb3B1cF8zMmJjMTBlODAzYzA0MTcwYjlmY2MzZmE5NTM5N2QxNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82ZDNjODgxNGY2OGQ0N2Q5YWI5ZDFjZDFiMWRjZDIyNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBjNzlkNDI2OTMyZDQ1YzM4MmRlZWJhZTNiOWFjM2I3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk1NTU0YjY4ZDhiOTRiYWJhNmQwY2MxMTljMTBmYjU1ID0gJCgnPGRpdiBpZD0iaHRtbF85NTU1NGI2OGQ4Yjk0YmFiYTZkMGNjMTE5YzEwZmI1NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVzaWduIEV4Y2hhbmdlLCBUb3JvbnRvIERvbWluaW9uIENlbnRyZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBjNzlkNDI2OTMyZDQ1YzM4MmRlZWJhZTNiOWFjM2I3LnNldENvbnRlbnQoaHRtbF85NTU1NGI2OGQ4Yjk0YmFiYTZkMGNjMTE5YzEwZmI1NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ZDNjODgxNGY2OGQ0N2Q5YWI5ZDFjZDFiMWRjZDIyNi5iaW5kUG9wdXAocG9wdXBfMGM3OWQ0MjY5MzJkNDVjMzgyZGVlYmFlM2I5YWMzYjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWUxMWEwMzNkZGJmNGUxZmIzMzY2NzBlYzJhY2RjYzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMmU4MmE5M2VjMWY0OGFmODEwODcyNmY1MmQ5YzU1ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ODQyZGIyM2Y1ZDA0MjcwYjM3MGRmODFmNjUwZWU5ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjg0MmRiMjNmNWQwNDI3MGIzNzBkZjgxZjY1MGVlOWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UyZTgyYTkzZWMxZjQ4YWY4MTA4NzI2ZjUyZDljNTVlLnNldENvbnRlbnQoaHRtbF82ODQyZGIyM2Y1ZDA0MjcwYjM3MGRmODFmNjUwZWU5ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZTExYTAzM2RkYmY0ZTFmYjMzNjY3MGVjMmFjZGNjMC5iaW5kUG9wdXAocG9wdXBfZTJlODJhOTNlYzFmNDhhZjgxMDg3MjZmNTJkOWM1NWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGQ1MTNjNTJmNjJjNGJhNGIwZWMxNjcxZTRiZjU0MWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMjUwMGI2ZjEwZGE0OWQ5YWI0ZTI3MzkzNzk2OTZkNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NTgwOGMwNDM1NTg0MDJkODEyNTM2OTNmYTdmNjVkMCA9ICQoJzxkaXYgaWQ9Imh0bWxfNzU4MDhjMDQzNTU4NDAyZDgxMjUzNjkzZmE3ZjY1ZDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTI1MDBiNmYxMGRhNDlkOWFiNGUyNzM5Mzc5Njk2ZDYuc2V0Q29udGVudChodG1sXzc1ODA4YzA0MzU1ODQwMmQ4MTI1MzY5M2ZhN2Y2NWQwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBkNTEzYzUyZjYyYzRiYTRiMGVjMTY3MWU0YmY1NDFhLmJpbmRQb3B1cChwb3B1cF9lMjUwMGI2ZjEwZGE0OWQ5YWI0ZTI3MzkzNzk2OTZkNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZWRjOTY0NGZkMTk0NjU1YmMyNjkzNWNkOTkwM2Y1ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsLTc5LjQxMTMwNzIwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhlNzZiZWIyMDNmMTRjZTBiMWJhOTJhMDdjODhiNjNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I2Y2Y2NWIzMTdmMTRiYzI4M2M2MmZmMDNhYjNiOWVhID0gJCgnPGRpdiBpZD0iaHRtbF9iNmNmNjViMzE3ZjE0YmMyODNjNjJmZjAzYWIzYjllYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rm9yZXN0IEhpbGwgTm9ydGgsIEZvcmVzdCBIaWxsIFdlc3QgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ZTc2YmViMjAzZjE0Y2UwYjFiYTkyYTA3Yzg4YjYzYi5zZXRDb250ZW50KGh0bWxfYjZjZjY1YjMxN2YxNGJjMjgzYzYyZmYwM2FiM2I5ZWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGVkYzk2NDRmZDE5NDY1NWJjMjY5MzVjZDk5MDNmNWUuYmluZFBvcHVwKHBvcHVwXzhlNzZiZWIyMDNmMTRjZTBiMWJhOTJhMDdjODhiNjNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1ZmMxMGQ1NDA2OTRjODY5NDczZmJiYTBiNzAxNGU1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzgxNTBmNDhlMWU3NDI0MThkN2ZmZjgxOTQxZGI4MGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTVlYTY0ZTlhNjdhNGEwN2I4YzZmZTA4NDI4YTlhYzUgPSAkKCc8ZGl2IGlkPSJodG1sX2E1ZWE2NGU5YTY3YTRhMDdiOGM2ZmUwODQyOGE5YWM1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc4MTUwZjQ4ZTFlNzQyNDE4ZDdmZmY4MTk0MWRiODBhLnNldENvbnRlbnQoaHRtbF9hNWVhNjRlOWE2N2E0YTA3YjhjNmZlMDg0MjhhOWFjNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NWZjMTBkNTQwNjk0Yzg2OTQ3M2ZiYmEwYjcwMTRlNS5iaW5kUG9wdXAocG9wdXBfNzgxNTBmNDhlMWU3NDI0MThkN2ZmZjgxOTQxZGI4MGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzFjZWUwMjk4ZGI0NDVjN2EzNTU5NDYyMTdjMmYzZDcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RkY2IxMDY2NWJkMjRlMDU4ZWU0N2U0OTUzNGE4MjFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EzMTNmNzAwMDJlYjRmZDdiZGVjYzZlZDQ5M2Y4NzEzID0gJCgnPGRpdiBpZD0iaHRtbF9hMzEzZjcwMDAyZWI0ZmQ3YmRlY2M2ZWQ0OTNmODcxMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm9yZCwgVW5pdmVyc2l0eSBvZiBUb3JvbnRvIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGRjYjEwNjY1YmQyNGUwNThlZTQ3ZTQ5NTM0YTgyMWQuc2V0Q29udGVudChodG1sX2EzMTNmNzAwMDJlYjRmZDdiZGVjYzZlZDQ5M2Y4NzEzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMxY2VlMDI5OGRiNDQ1YzdhMzU1OTQ2MjE3YzJmM2Q3LmJpbmRQb3B1cChwb3B1cF9kZGNiMTA2NjViZDI0ZTA1OGVlNDdlNDk1MzRhODIxZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jYmNkMjE3YWViYTA0ZWI5YmVhZTc5YWNhNjU0YjFlZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmRhYzlkMTg4ODkzNDhlNmJmMjFlZGE0M2U0OTcxNDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzBjMGYzMjY4YjJmNDg2OWFlY2RlYmQxYTZiMWQ3ZjQgPSAkKCc8ZGl2IGlkPSJodG1sXzMwYzBmMzI2OGIyZjQ4NjlhZWNkZWJkMWE2YjFkN2Y0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaGluYXRvd24sIEdyYW5nZSBQYXJrLCBLZW5zaW5ndG9uIE1hcmtldCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJkYWM5ZDE4ODg5MzQ4ZTZiZjIxZWRhNDNlNDk3MTQyLnNldENvbnRlbnQoaHRtbF8zMGMwZjMyNjhiMmY0ODY5YWVjZGViZDFhNmIxZDdmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jYmNkMjE3YWViYTA0ZWI5YmVhZTc5YWNhNjU0YjFlZS5iaW5kUG9wdXAocG9wdXBfMmRhYzlkMTg4ODkzNDhlNmJmMjFlZGE0M2U0OTcxNDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjAzNmJlNTg2NjI2NDg4NGI1M2I5ZjY2M2I2OTQ1ODAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcwYjU2MDg5NTQ1ZTQzZTM5MzBhNDc4ZTViNTFmM2I1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk0NjgyOGViYTdiNDQxNmNiODBiODZlMTIyYjgxY2IxID0gJCgnPGRpdiBpZD0iaHRtbF85NDY4MjhlYmE3YjQ0MTZjYjgwYjg2ZTEyMmI4MWNiMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q04gVG93ZXIsIEJhdGh1cnN0IFF1YXksIElzbGFuZCBhaXJwb3J0LCBIYXJib3VyZnJvbnQgV2VzdCwgS2luZyBhbmQgU3BhZGluYSwgUmFpbHdheSBMYW5kcywgU291dGggTmlhZ2FyYSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcwYjU2MDg5NTQ1ZTQzZTM5MzBhNDc4ZTViNTFmM2I1LnNldENvbnRlbnQoaHRtbF85NDY4MjhlYmE3YjQ0MTZjYjgwYjg2ZTEyMmI4MWNiMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMDM2YmU1ODY2MjY0ODg0YjUzYjlmNjYzYjY5NDU4MC5iaW5kUG9wdXAocG9wdXBfNzBiNTYwODk1NDVlNDNlMzkzMGE0NzhlNWI1MWYzYjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjI0ZmE1YTg4MTBiNDBhOGI5ZDA1Nzg3MjJiOTkyODIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZjBmNTFkN2U5YTI0M2JkOWIzNzFmMWM4OWZlNGM1ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOGU3OTY1Y2Y4Nzc0MGM1YjlkMTE2MDE0N2Q0ZjJmOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzhlNzk2NWNmODc3NDBjNWI5ZDExNjAxNDdkNGYyZjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIDI1IFRoZSBFc3BsYW5hZGUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ZjBmNTFkN2U5YTI0M2JkOWIzNzFmMWM4OWZlNGM1ZS5zZXRDb250ZW50KGh0bWxfMzhlNzk2NWNmODc3NDBjNWI5ZDExNjAxNDdkNGYyZjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjI0ZmE1YTg4MTBiNDBhOGI5ZDA1Nzg3MjJiOTkyODIuYmluZFBvcHVwKHBvcHVwXzhmMGY1MWQ3ZTlhMjQzYmQ5YjM3MWYxYzg5ZmU0YzVlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I3YjExNmQ1ZjA3MjQ0YTQ4YWE5ZmQyZDJhZjdjMDQwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM4ZjAwMDI1MTMzODQyNzU4OTZmMTE3N2RiNTVhZTcwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ExYjdmNDNjMDJhNTQ5YWZiOTJiZTE5OGVkY2MzZWRmID0gJCgnPGRpdiBpZD0iaHRtbF9hMWI3ZjQzYzAyYTU0OWFmYjkyYmUxOThlZGNjM2VkZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOGYwMDAyNTEzMzg0Mjc1ODk2ZjExNzdkYjU1YWU3MC5zZXRDb250ZW50KGh0bWxfYTFiN2Y0M2MwMmE1NDlhZmI5MmJlMTk4ZWRjYzNlZGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjdiMTE2ZDVmMDcyNDRhNDhhYTlmZDJkMmFmN2MwNDAuYmluZFBvcHVwKHBvcHVwXzM4ZjAwMDI1MTMzODQyNzU4OTZmMTE3N2RiNTVhZTcwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZmZTQxN2U2Njc3NjQ2MDViOTcyZWExOWNhMGQ4OTMwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDQ5NTk0Mzg3YzU2NGYxMmJlODlhMDc2OTE5Mzg1ZmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2EzNGZmMmVlOGViNGMxZWI1YjJhOWYyNjMyMDZjNTcgPSAkKCc8ZGl2IGlkPSJodG1sXzdhMzRmZjJlZThlYjRjMWViNWIyYTlmMjYzMjA2YzU3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3ZlcmNvdXJ0IFZpbGxhZ2UsIER1ZmZlcmluIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDQ5NTk0Mzg3YzU2NGYxMmJlODlhMDc2OTE5Mzg1ZmQuc2V0Q29udGVudChodG1sXzdhMzRmZjJlZThlYjRjMWViNWIyYTlmMjYzMjA2YzU3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZmZTQxN2U2Njc3NjQ2MDViOTcyZWExOWNhMGQ4OTMwLmJpbmRQb3B1cChwb3B1cF8wNDk1OTQzODdjNTY0ZjEyYmU4OWEwNzY5MTkzODVmZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NTUxMGQ5MDkwODM0YzM3YTFjNjBmMDJlNjQzZjM1OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZWQyODVjZDVjMzg0NjNiOTI4ODE1YTgxMTdiNTQ1NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMmE5NmIzMDM0NTQ0MzhlYTRkZDliOWEyNzI1Nzg0YyA9ICQoJzxkaXYgaWQ9Imh0bWxfMDJhOTZiMzAzNDU0NDM4ZWE0ZGQ5YjlhMjcyNTc4NGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FlZDI4NWNkNWMzODQ2M2I5Mjg4MTVhODExN2I1NDU3LnNldENvbnRlbnQoaHRtbF8wMmE5NmIzMDM0NTQ0MzhlYTRkZDliOWEyNzI1Nzg0Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NTUxMGQ5MDkwODM0YzM3YTFjNjBmMDJlNjQzZjM1OS5iaW5kUG9wdXAocG9wdXBfYWVkMjg1Y2Q1YzM4NDYzYjkyODgxNWE4MTE3YjU0NTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzk5Yjk0MjcxNTFmNDQzM2JmMTYyMDcwNGE5NDJmYTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZGZiOTZhZDk0NjA0YTc2YWQyMTQzM2ZhMTE3N2Y3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMzNlMTk0MmU5OGI0YTJjOTdhMzg3ODQ4YmViODRmNyA9ICQoJzxkaXYgaWQ9Imh0bWxfYTMzZTE5NDJlOThiNGEyYzk3YTM4Nzg0OGJlYjg0ZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBFeGhpYml0aW9uIFBsYWNlLCBQYXJrZGFsZSBWaWxsYWdlIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2RmYjk2YWQ5NDYwNGE3NmFkMjE0MzNmYTExNzdmN2Euc2V0Q29udGVudChodG1sX2EzM2UxOTQyZTk4YjRhMmM5N2EzODc4NDhiZWI4NGY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M5OWI5NDI3MTUxZjQ0MzNiZjE2MjA3MDRhOTQyZmE1LmJpbmRQb3B1cChwb3B1cF83ZGZiOTZhZDk0NjA0YTc2YWQyMTQzM2ZhMTE3N2Y3YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNmE5NjJkODRjYzQ0YTFhODI1NmQxOTZmZmRhZWI3YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MTYwODMsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VjNDJmYmVjYTQyODQ3MTdiMjA2NWI3Yzg2Yjc4ZmMxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzViZGE2MDY5NGEzZjRlMjE4YjNmN2NmZDc1NTExYmRlID0gJCgnPGRpdiBpZD0iaHRtbF81YmRhNjA2OTRhM2Y0ZTIxOGIzZjdjZmQ3NTUxMWJkZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlnaCBQYXJrLCBUaGUgSnVuY3Rpb24gU291dGggQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYzQyZmJlY2E0Mjg0NzE3YjIwNjViN2M4NmI3OGZjMS5zZXRDb250ZW50KGh0bWxfNWJkYTYwNjk0YTNmNGUyMThiM2Y3Y2ZkNzU1MTFiZGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTZhOTYyZDg0Y2M0NGExYTgyNTZkMTk2ZmZkYWViN2IuYmluZFBvcHVwKHBvcHVwX2VjNDJmYmVjYTQyODQ3MTdiMjA2NWI3Yzg2Yjc4ZmMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U3ODUwNWQ0YjdiYTQ4MzRiMmZiYjUyNzkzZDQ2NTAwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYwOTE0MmY4NTA2ZjQzMjY5M2VmMWYzMmJmMjE2Njk1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZhNDJiMGZmMWIyZjQwYTliMmI4Y2U4M2MyODcxNWUzID0gJCgnPGRpdiBpZD0iaHRtbF82YTQyYjBmZjFiMmY0MGE5YjJiOGNlODNjMjg3MTVlMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcyBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYwOTE0MmY4NTA2ZjQzMjY5M2VmMWYzMmJmMjE2Njk1LnNldENvbnRlbnQoaHRtbF82YTQyYjBmZjFiMmY0MGE5YjJiOGNlODNjMjg3MTVlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNzg1MDVkNGI3YmE0ODM0YjJmYmI1Mjc5M2Q0NjUwMC5iaW5kUG9wdXAocG9wdXBfNjA5MTQyZjg1MDZmNDMyNjkzZWYxZjMyYmYyMTY2OTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGUxOGQ4OTJjNTdmNGQ4ZGJiNjJhOWM2ZmY3M2M2Y2YgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmMDQyODQ3NTFlZDQ1MjFiM2U3Mjg2Nzg0Y2E4MWZmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg1YjBiZjkwZDJjODQ3NTA4MThjNTcyMjk5ZjQ0MjBmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YxODdhZGFjYTNlNjQxZGZiMTQ1ZTU2YjBkZWY1Mzk2ID0gJCgnPGRpdiBpZD0iaHRtbF9mMTg3YWRhY2EzZTY0MWRmYjE0NWU1NmIwZGVmNTM5NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODViMGJmOTBkMmM4NDc1MDgxOGM1NzIyOTlmNDQyMGYuc2V0Q29udGVudChodG1sX2YxODdhZGFjYTNlNjQxZGZiMTQ1ZTU2YjBkZWY1Mzk2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RlMThkODkyYzU3ZjRkOGRiYjYyYTljNmZmNzNjNmNmLmJpbmRQb3B1cChwb3B1cF84NWIwYmY5MGQyYzg0NzUwODE4YzU3MjI5OWY0NDIwZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MGQxYzcyZTNjZmQ0NzA3ODE1M2ZmYjI5YzZkOTY2MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjMwMTUsLTc5LjM4OTQ5MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGYwNDI4NDc1MWVkNDUyMWIzZTcyODY3ODRjYTgxZmYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjRlMWIyZThkZDI0NDExOGFjNTA1NDcxMTk4MzJhN2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmQ0MjAzMjA4MzU5NDg5ZDgwMTBkNjFmMTFlYzI5YTggPSAkKCc8ZGl2IGlkPSJodG1sXzZkNDIwMzIwODM1OTQ4OWQ4MDEwZDYxZjExZWMyOWE4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RdWVlbiYjMzk7cyBQYXJrIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjRlMWIyZThkZDI0NDExOGFjNTA1NDcxMTk4MzJhN2Iuc2V0Q29udGVudChodG1sXzZkNDIwMzIwODM1OTQ4OWQ4MDEwZDYxZjExZWMyOWE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUwZDFjNzJlM2NmZDQ3MDc4MTUzZmZiMjljNmQ5NjYyLmJpbmRQb3B1cChwb3B1cF82NGUxYjJlOGRkMjQ0MTE4YWM1MDU0NzExOTgzMmE3Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNjYyOGNmNDUyYjI0NjlhOGQxZGJiZGJmNjNiNTZkOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Mjc0MzksLTc5LjMyMTU1OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjA0Mjg0NzUxZWQ0NTIxYjNlNzI4Njc4NGNhODFmZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMWE5N2NjNTc0NTg0MmRmOTQ5NWNjNDdiODEwZDk0MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNzY2OGE0MDFjZTE0NDQ2YWZmMzU3MWZjNGMwNjVmNyA9ICQoJzxkaXYgaWQ9Imh0bWxfYTc2NjhhNDAxY2UxNDQ0NmFmZjM1NzFmYzRjMDY1ZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJ1c2luZXNzIFJlcGx5IE1haWwgUHJvY2Vzc2luZyBDZW50cmUgOTY5IEVhc3Rlcm4gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMWE5N2NjNTc0NTg0MmRmOTQ5NWNjNDdiODEwZDk0MC5zZXRDb250ZW50KGh0bWxfYTc2NjhhNDAxY2UxNDQ0NmFmZjM1NzFmYzRjMDY1ZjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTY2MjhjZjQ1MmIyNDY5YThkMWRiYmRiZjYzYjU2ZDguYmluZFBvcHVwKHBvcHVwX2MxYTk3Y2M1NzQ1ODQyZGY5NDk1Y2M0N2I4MTBkOTQwKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



I tried a number of different values for K, but nothing really resulted in greater distribution. It may be more fruitful to look at Greater Toronto as a whole, but that was beyond the scope of the assignment. 



```python

```
