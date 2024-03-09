import geopandas as gpd
import pandas
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def displayFile(dataframe, removed_rows, output, feature):
    # Plot language data
    ax = dataframe.plot(column=feature, cmap='Greens', linewidth=0.1, edgecolor='0.8', legend=True)

    # Plot removed districts where no data is present with grey color
    if not removed_rows.empty:
        removed_rows.plot(ax=ax, color='grey', linewidth=0.1, edgecolor='0.8', alpha=0.5, label='No data')

    # Hide axes
    ax.axis('off')

    # Create legend and add to plot
    legend_elements = [Patch(facecolor='grey', edgecolor='grey', alpha=0.5, label='No data')]
    ax.legend(handles=legend_elements, loc='lower left')

    # Set DPI(dots per inch) to get high resolution image and save to specified directory
    plt.savefig(output, dpi=800, bbox_inches='tight')

    # Show plot
    plt.show()


def processFile(shapeDF, speakersDF):
    # get GUIDs from speakersDF
    unique_guids = set(speakersDF['GUID'])

    # Filter shapeDF to remove rows that are not present in speakersDF (no data)
    filtered_shapeDF = shapeDF[shapeDF['GUID'].isin(unique_guids)]

    removed_rows = shapeDF[~shapeDF['GUID'].isin(unique_guids)]
    # for index, row in removed_rows.iterrows():
    #     print("Removed row:", row)

    # Merge speakersDF with filtered_shapeDF on GUID
    merged_df = filtered_shapeDF.merge(speakersDF, on='GUID', how='left')

    return merged_df, removed_rows

shapeDF = gpd.read_file(r".\shapefiles\Galway_ED_20M.geojson")
speakersDF = pandas.read_csv('Regressor_Prediction.csv')
dataframe, removed_rows = processFile(shapeDF, speakersDF)
for index, row in dataframe.iterrows():
    print(row)
displayFile(dataframe,removed_rows, r'.\maps\output_map.png',"Speakers2022")
