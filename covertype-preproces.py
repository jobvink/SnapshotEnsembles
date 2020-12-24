import pandas as pd

X_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
       'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area']

y_col = 'Cover_Type'

data = pd.read_csv('data/covtype.data')
data.reset_index()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(data[X_cols])
X = pd.DataFrame(X, columns=X_cols)

y = pd.DataFrame(data[y_col].values, columns=[y_col])

labels = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz',
}

y = y.replace(labels)

data = pd.concat([X, y], axis=1)

data.to_csv('data/covtype-normalized.csv', index=False)