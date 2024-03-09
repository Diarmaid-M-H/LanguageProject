import geopandas as gpd
from matplotlib import pyplot as plt


def displayFile(dataframe,output):
    # Read the GeoJSON file into a GeoDataFrame
    # Plot the GeoDataFrame
    dataframe.plot(color='green')
    plt.axis('off')
    # Set DPI(dots per inch) to get high resolution image and save it to specified directory
    plt.savefig(output, dpi=800, bbox_inches='tight')
    # Show the plot
    plt.show()

def processFile(shapefile,speakers):
    gdf = gpd.read_file(shapefile)
    return gdf


dataframe = processFile(r".\shapefiles\Galway_ED_20M.geojson","foobar")
displayFile(dataframe,r'.\maps\output_map.png')
