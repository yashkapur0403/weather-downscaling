// OPTIONAL: Google Earth Engine alternative path (only if the direct downloads
// in download_or_export.py fail for you). Paste into:
//     https://code.earthengine.google.com
//
// Exports to your Google Drive:
//   1. Daily CHIRPS 0.05-deg rainfall over the ROI, one GeoTIFF per day,
//      named CHIRPS_YYYY-MM-DD.tif
//   2. SRTM DEM 30 m over the ROI, named DEM_roi.tif
//
// After exports finish, download from Drive into:
//   data/raw/chirps/   (CHIRPS_*.tif)
//   data/raw/dem/      (DEM_roi.tif)
//
// preprocess.py auto-detects these files and converts them to the internal
// chirps_roi.npz / dem_roi.npz formats, so the rest of the pipeline is
// unchanged either way.

// ---------------- configure ----------------
var roi = ee.Geometry.Rectangle([73.0, 12.0, 78.0, 17.0]); // [lonMin, latMin, lonMax, latMax]
var start = '2022-06-01';
var end   = '2022-08-31';

// ---------------- 1. daily CHIRPS ----------------
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(start, end)
    .filterBounds(roi);

var days = chirps.size().getInfo();
print('CHIRPS daily images found:', days);

var daily = chirps.map(function (img) {
  return img.select('precipitation').clip(roi)
      .set('system:date', img.date().format('YYYY-MM-dd'));
});

var listOfDates = daily.aggregate_array('system:date');
listOfDates.evaluate(function (dates) {
  dates.forEach(function (d) {
    var img = daily.filterDate(d, ee.Date(d).advance(1, 'day')).first();
    Export.image.toDrive({
      image: img,
      description: 'CHIRPS_' + d,
      folder: 'chirps_export',
      region: roi,
      scale: 5566,           // ~0.05 deg at these latitudes
      crs: 'EPSG:4326',
      maxPixels: 1e13
    });
  });
});

// ---------------- 2. SRTM DEM ----------------
var dem = ee.Image('USGS/SRTMGL1_003').clip(roi);
Export.image.toDrive({
  image: dem,
  description: 'DEM_roi',
  folder: 'dem_export',
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
