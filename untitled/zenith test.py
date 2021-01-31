import pvlib
from pandas import datetime
print(pvlib.solarposition.get_solarposition(datetime.now(), 55, -2, altitude=200))