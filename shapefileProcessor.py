import geopandas as gpd
import pandas
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def displayFile(dataframe, removed_rows, output, feature):
    # Plot language data
    ax = dataframe.plot(column=feature, cmap='Blues', linewidth=0.1, edgecolor='0.8', legend=True, vmin=0.0, vmax=0.5)

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


def displayChange(dataframe, removed_rows, output, featureBefore, featureAfter):
    # Calculate the difference between featureAfter and featureBefore
    dataframe['difference'] = dataframe[featureAfter] - dataframe[featureBefore]

    # Plot the difference column
    ax = dataframe.plot(column='difference', cmap='RdYlGn', linewidth=0.1, edgecolor='0.8', legend=True,
                        vmin=-0.5, vmax=0.5)
    # Create legend and add to plot
    legend_elements = [Patch(facecolor='grey', edgecolor='grey', alpha=0.5, label='No data')]

    # Plot removed districts where no data is present with grey color
    if not removed_rows.empty:
        removed_rows.plot(ax=ax, color='grey', linewidth=0.1, edgecolor='0.8', alpha=0.5, label='No data')

    # Hide axes
    ax.axis('off')

    # Create legend and add to plot
    ax.legend(handles=legend_elements, loc='lower left')

    # Set DPI(dots per inch) to get high resolution image and save to specified directory
    plt.savefig(output, dpi=800, bbox_inches='tight')

    # Show plot
    plt.show()


def displayHistoricalChange():
    displayChange(dataframe, removed_rows, r'.\maps\final2006_2011', "Speakers2006", "Speakers2011")
    displayChange(dataframe, removed_rows, r'.\maps\final2011_2016', "Speakers2011", "Speakers2016")
    displayChange(dataframe, removed_rows, r'.\maps\final2016_2022', "Speakers2016", "Speakers2022")


shapeDF = gpd.read_file(r".\shapefiles\Galway_ED_20M.geojson")
#speakersDF = pandas.read_csv(r'abilityToSpeakIrish.csv')
#speakersDF = pandas.read_csv(r'RegressorPredictionWithError.csv')
speakersDF = pandas.read_csv(r'TunedEpidemiology.csv')
dataframe, removed_rows = processFile(shapeDF, speakersDF)
# for index, row in dataframe.iterrows():
#     print(row)
#displayChange(dataframe, removed_rows,r'.\maps\Crungle.png',"Speakers2022","Speakers2027")
# displayFile(dataframe, removed_rows, r'.\maps\final2006',"Speakers2006")
# displayFile(dataframe, removed_rows, r'.\maps\final2011',"Speakers2011")
# displayFile(dataframe, removed_rows, r'.\maps\final2016',"Speakers2016")
# displayFile(dataframe, removed_rows, r'.\maps\final2022',"Speakers2022")

# displayFile(dataframe, removed_rows, r'.\maps\IrishSpeakers2006',"Speakers2006")
# displayFile(dataframe, removed_rows, r'.\maps\IrishSpeakers2011',"Speakers2011")
# displayFile(dataframe, removed_rows, r'.\maps\IrishSpeakers2016',"Speakers2016")
# displayFile(dataframe, removed_rows, r'.\maps\IrishSpeakers2022',"Speakers2022")

#displayFile(dataframe,removed_rows,r'.\maps\Crungle.png',"Speakers2027")
displayFile(dataframe,removed_rows,r'.\maps\EpidemiologyTunedError.png',"Error")
#displayFile(dataframe,removed_rows,r'.\maps\EpidemiologyTunedPrediction',"Speakers2027")
#displayChange(dataframe,removed_rows,r'.\maps\EpidemiologyTunedChangePredicted',"Speakers2022","Speakers2027")
#displayHistoricalChange()
