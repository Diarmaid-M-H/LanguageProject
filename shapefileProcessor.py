import geopandas as gpd
import pandas
from matplotlib import pyplot as plt



def displayFile(dataframe, output, feature):
    # Plot the GeoDataFrame with specified feature column
    dataframe.plot(column=feature, cmap='Greens', linewidth=0.8, edgecolor='0.8', legend=True)
    plt.axis('off')
    # Set DPI(dots per inch) to get high resolution image and save it to specified directory
    plt.savefig(output, dpi=800, bbox_inches='tight')
    # Show the plot
    plt.show()


def processFile(shapeDF, speakersDF):
    # Create a set of unique GUIDs from speakersDF
    unique_guids = set(speakersDF['GUID'])

    # Filter shapeDF to remove rows with GUID not present in speakersDF (coos, loughatorick, etc)
    filtered_shapeDF = shapeDF[shapeDF['GUID'].isin(unique_guids)]

    # removed_rows = shapeDF[~shapeDF['GUID'].isin(unique_guids)]
    # for index, row in removed_rows.iterrows():
    #     print("Removed row:", row)

    # Merge speakersDF with filtered_shapeDF on GUID
    merged_df = filtered_shapeDF.merge(speakersDF, on='GUID', how='left')

    return merged_df

shapeDF = gpd.read_file(r".\shapefiles\Galway_ED_20M.geojson")
speakersDF = pandas.read_csv('Regressor_Prediction.csv')
dataframe = processFile(shapeDF, speakersDF)
for index, row in dataframe.iterrows():
    print(row)
displayFile(dataframe, r'.\maps\output_map.png',"Speakers2022")
