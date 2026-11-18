# Data access and redistribution

The source archive references GCM driving fields (CNRM-ESM2-1), regional RCM simulation output (CNRM-ALADIN), radiative forcing time series, an RCM land mask, coarse-orography and high-resolution topographic inputs based on Copernicus DEM, and an external U-Net CMIP-CMIP benchmark output. None of the raw climate datasets or model weights is included in this code archive. A 2 KB derived coarse-orography GCM array is included solely as an author-provided preparation input; its right to redistribution and provenance must be confirmed before public release.

Input placeholders/path keys can be inspected in `configs/config_model_E.json` and the data-preparation source. Original absolute CEAZA/workspace paths are implementation details, **not** public download links. The release requires an author-approved statement detailing what is public, what requires permission, and how users may reconstruct the same data. Do not assert availability of data whose distribution rights have not been checked.

Any shared trained weights or large prediction cubes should be versioned separately (e.g., an authorized data repository or release assets when licensed) and identified by checksum. No public hosting location has been verified here.
