# The Sentinels EOPF Toolkit

<p align="center"><img title="EOPF Toolkit Logo" alt="Alt text" src="/img/EOPF_FINAL_LOGO.png" class="center" width=40%></p>
<p align="center"><img title="Consortium logos" alt="Alt text" src="/img/logos.png" class="center" width=80%></p>

## About
EOPF 101 is a community-driven toolkit that facilitates the adoption of the Zarr data format for Copernicus Sentinel data, targeting users who are new to cloud computing. The [EOPF Toolkit project](https://github.com/eopf-toolkit) is developed by [Development Seed](https://developmentseed.org/), [thriveGEO](https://thrivegeo.com/) and [Sparkgeo](https://sparkgeo.com/), together with a group of champion users. Together they are creating EOPF 101, a user-friendly resource consisting of documentation, Jupyter Notebooks and plug-ins that showcase the use of Zarr format Sentinel data for applications across multiple domains.

The Sentinels EOPF Toolkit is a project funded by the [European Space Agency](https://www.esa.int/).

The toolkit consists of a set of community resources, including an online book, community libraries and plugins, thematic case studies and a notebook competition:

### EOPF 101 - Outline
* Chapter 1 - About EOPF
* Chapter 2 - About EOPF Zarr
* Chapter 3 - About Chunking
* Chapter 4 - EOPF and STAC
* Chapter 5 - Tools to work with EOPF Zarr
* Chapter 6 - EOPF Zarr in Action
* Glossary
* References

### Overview of libraries and plugins
We will develop a series of open-source libraries and plugins, including the following:
* **Explore Zarr in STAC**: Pystac and QGIS usae with the EOPF STAC catalogue
* **Stackstac**
* **R with Rarr**
* **GDAL** evolution of the current driver
* **Titiler-multidimensional**: Prepare a docker for starting a titiler tailored for EOPF Zarr

### Case studies
Together with a group of champion users, we will developed and publish technical and thematic case studies that include example Jupyter Notebook workflows for using Sentinels data in Zarr format:
* `Technical case studies`:
  * Access the EOPF Zarr STAC API with R
  * Access the EOPF Zarr STAC API with QGIS
  * Access and analyse EOPF STAC Zarr data with R
  * Raster cube with rstac
  * Create and Visualise Zarr at Several Resolutions (Creating Zarr Overviews, Visualising Multiscale Pyramids)
* `Thematic case studies`:
  * Flood Mapping - Time Series Analysis in Valencia
  * Surface Water Dynamics - Time Series Analysis with Sentinel-1
  * Fire in Sardinia 2025 (True and False Composites with Sentinel-2, Heat Detection with Sentinel-3 LST, Normalised Burn Ratio)
  * Reservoir Surface Monitoring
  * Analysing Forest Vegetation Anomalies
  * African rangeland monitoring with Sentinel-2 and Sentinel-3

### Notebook competition
Between October 2025 and March 2026, we ran a notebook competition inviting Sentinel data users to work with the live sample data reprocessed as part of the [EOPF Sentinel Zarr Sample Service](https://zarr.eopf.copernicus.eu/). We received 12 submissions. A total of five winners were selected and will be integrated into EOPF 101.

## Clean notebooks

To ensure smooth CI/CD operations and prevent build failures, all Jupyter notebooks in this repository must be kept "clean" (without outputs or execution counts). Large notebooks with embedded images or outputs can cause buffer overflow errors during the build process.

### How to Clean Notebooks

#### Method 1: Using nbstripout (Recommended)

```bash
# Install nbstripout
pip install nbstripout

# Clean a single notebook
nbstripout notebook.ipynb

# Clean all notebooks
nbstripout *.ipynb

# Check if notebooks are clean (exits with error if not clean)
nbstripout --verify *.ipynb
```

#### Method 2: Using Jupyter Interface

1. Open the notebook in Jupyter Lab/Notebook
2. Go to `Kernel` → `Restart & Clear Output`
3. Save the notebook
4. Commit the changes

#### Method 3: Command Line Alternative
```bash
# Single notebook
jupyter nbconvert --clear-output --inplace notebook.ipynb

# All notebooks
find . -name '*.ipynb' -not -path './_book/*' \
  -exec nbstripout {} +
```

## Development timeline
By March 2026, we will have develped a community resource where you can learn how to use the EOPF Sentinel Zarr Samples Service by ESA. It is designed for Sentinel data users who are new to cloud computing.

* `June 2025`: Launch of first version during Living Planet Symposium
* `2nd half of 2025`: Development of thematic case studies together with champion users
* `Oct 2025 to Mar 2026`: EOPF Notebook competition
* `throughout`: Communications and outreach through social media and conference presence

## Get involved
We welcome to join you in this community effort in the following ways:
* Follow us here on [Github](https://github.com/eopf-toolkit)

## Install dependencies
We use uv to install the packages and dependencies. If you want to recreatre this environment run the following command.
```bash
uv sync
```
