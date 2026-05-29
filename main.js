
//------------------------------------------------------------------------------
// 0. DEFINIZIONE ROI E PARAMETRI DI ANALISI
//------------------------------------------------------------------------------
var vigneti_fc = ee.FeatureCollection(vigneto);
var uliveti_fc = ee.FeatureCollection(uliveto);
//var roi = uliveti_fc;
var roi = vigneti_fc;
Map.centerObject(roi, 15);
Map.addLayer(vigneto, {color: 'purple'}, 'Il Mio Vigneto');
Map.addLayer(uliveto, {color: 'green'},   'Il Mio Uliveto');

var startYear     = 2017;
var endYear       = 2024;

var calculateAndPrintROIStats = function(roi) {
  // Calcola l'area in metri quadrati. Il parametro '1' è l'errore massimo consentito.
  var area_sq_m = roi.geometry().area(1);
  // Converte l'area in ettari (1 ettaro = 10,000 m^2).
  var area_ha = area_sq_m.divide(10000);
  
  // Per contare i pixel, creiamo un'immagine costante e usiamo reduceRegion.
  var pixel_count = ee.Image.constant(1).reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: roi.geometry(),
    scale: 10, // Scala coerente con i dati Sentinel (10m).
    maxPixels: 1e13
  }).get('constant'); // Il risultato è in un dizionario con la chiave del nome della banda.

  // Usa .evaluate() per recuperare i risultati dal server in modo asincrono
  // e stamparli nella console del browser.
  ee.Dictionary({area: area_ha, pixels: pixel_count}).evaluate(function(results, error) {
    if (error) {
      print('Errore nel calcolo delle statistiche:', error);
    } else {
      print('Statistiche ROI:',
            'Superficie: ' + results.area.toFixed(2) + ' ettari',
            'Numero di pixel (a 10m): ' + results.pixels);
    }
  });
};

// Esegui la funzione per ciascuna ROI.
calculateAndPrintROIStats(roi);


//------------------------------------------------------------------------------
// 1. PREPARAZIONE DATI SENTINEL-1 CON INDICI
//------------------------------------------------------------------------------

var analysisBands = ['VV', 'VH', 'RVI', 'RFDI'];

//LEE ANTI-SPECKLE FILTER
function applyLeeFilter(image) {
  var vv = ee.Image(image).select('VV'),
      vh = ee.Image(image).select('VH');

  // Convert dB to linear
  var vvLin = ee.Image(10).pow(vv.divide(10)),
      vhLin = ee.Image(10).pow(vh.divide(10));

  var kernel = ee.Kernel.square({radius: 1});

  function leeOneBand(bandLin, name) {
    var mean = bandLin.reduceNeighborhood(ee.Reducer.mean(), kernel);
    var variance = bandLin.reduceNeighborhood(ee.Reducer.variance(), kernel);
    var cv = variance.sqrt().divide(mean);
    var k = cv.multiply(cv).divide(cv.multiply(cv).add(0.25));
    var filtered = mean.add(k.multiply(bandLin.subtract(mean)));
    // back to dB
    return ee.Image(10).multiply(filtered.log10()).rename(name);
  }

  var vvF = leeOneBand(vvLin, 'VV'),
      vhF = leeOneBand(vhLin, 'VH');

  return image.addBands([vvF, vhF], null, true);
}


function calculateRVI(image) {
  var vh = image.select('VH');
  var vv = image.select('VV');
  
  var rvi = vv.divide(vv.add(vh)).sqrt()
               .multiply(vv.divide(vh)).rename('RVI');
  return image.addBands(rvi);
}

function calculateRFDI(image) {
  var vvLin = ee.Image(10).pow(image.select('VV').divide(10));
  var vhLin = ee.Image(10).pow(image.select('VH').divide(10));
  var rfdi = vvLin.subtract(vhLin)
                  .divide(vvLin.add(vhLin))
                  .rename('RFDI');
  return image.addBands(rfdi);
}

function addAllIndices(image) {
  // Apply speckle filter
  image = applyLeeFilter(image);
  image = calculateRVI(image);
  image = calculateRFDI(image);
  var date = ee.Date(image.get('system:time_start'))
                 .format('YYYY-MM-dd HH:mm:ss');
  return image.set('datetime', date);
}

var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(roi)
  .filter(ee.Filter.date(ee.Date.fromYMD(startYear, 1, 1), ee.Date.fromYMD(endYear, 12, 31)))
  .filterMetadata('transmitterReceiverPolarisation','equals',['VV','VH'])
  .filterMetadata('instrumentMode','equals','IW')
  .filter(ee.Filter.eq('orbitProperties_pass','ASCENDING'))
  .map(addAllIndices);

//------------------------------------------------------------------------------
// 2. CREAZIONE COMPOSITO MEDIANO PER L'ANALISI
//------------------------------------------------------------------------------
var medianaTemporale = s1.select(analysisBands).median().clip(roi);
Map.addLayer(medianaTemporale, {bands: ['VV', 'VH', 'RVI'], min: [-20, -25, 0], max: [0, -5, 1]}, 'Composito Mediano (VV/VH/RVI)', false);


//------------------------------------------------------------------------------
// 3. ANALISI DELLE COMPONENTI PRINCIPALI (PCA)
//------------------------------------------------------------------------------
var scale = 30;

// 3.1 Normalizzazione (Z-Score)
var mediaSpaziale = medianaTemporale.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: scale,
    maxPixels: 1e9
});
var stdDevsSpaziale = medianaTemporale.reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: roi,
    scale: scale,
    maxPixels: 1e9
});
var normalizedImage = medianaTemporale.toArray()
  .subtract(ee.Image(ee.Array(mediaSpaziale.values(analysisBands))))
  .divide(ee.Image(ee.Array(stdDevsSpaziale.values(analysisBands))))
  .arrayProject([0])
  .arrayFlatten([analysisBands]);

// 3.2 Calcolo della Matrice di Covarianza
var arrayImage = normalizedImage.select(analysisBands).toArray(); //lo rendiamo nuovamente single band "array"
var covarianceMatrix = arrayImage.reduceRegion({
  reducer: ee.Reducer.covariance(),
  geometry: roi,
  scale: scale,
  maxPixels: 1e9
});
var covArray = ee.Array(covarianceMatrix.get('array'));

// 3.3 Calcolo Autovettori e Autovalori
var eigens = covArray.eigen();
var eigenvalues = eigens.slice(1, 0, 1).project([0]);
var eigenvectors = eigens.slice(1, 1);

// 3.4 Proiezione sui PC
var imageArray1D = normalizedImage.select(analysisBands).toArray().toArray(1);
var pcImage = ee.Image(eigenvectors)
  .matrixMultiply(imageArray1D)
  .arrayProject([0])
  .arrayFlatten([['PC1','PC2','PC3','PC4']]);


//------------------------------------------------------------------------------
// 4. VISUALIZZAZIONE RISULTATI PCA
//------------------------------------------------------------------------------
print('--- Risultati PCA ---');

// 4.1 Scree Plot (Varianza spiegata da ogni PC)
var varianceExplained = eigenvalues.divide(eigenvalues.reduce(ee.Reducer.sum(), [0]).get([0]))
  .multiply(100);
  
// Trasferiamo i risultati lato client per usarli nei titoli dei grafici
var varianceExplainedList = varianceExplained.getInfo();

print('Autovalori:', eigenvalues);
print('Autovettori:', eigenvectors);
print('Varianza Spiegata (%):', varianceExplained);

var screePlot = ui.Chart.array.values(varianceExplained, 0, ee.List.sequence(1, analysisBands.length).map(function(n){return ee.String('PC').cat(ee.Number(n).int())}))
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Scree Plot - Varianza Spiegata per Componente',
    hAxis: {title: 'Componente Principale'},
    vAxis: {title: '% Varianza Spiegata'},
    legend: {position: 'none'}
  });
print(screePlot);


// 4.2 Grafico di Dispersione dei Punteggi (Score Plot)
var sample = pcImage.select(['PC1', 'PC2']).sample({
  region: roi,
  scale: scale,
  numPixels: 500,
  geometries: true
});

var scorePlot = ui.Chart.feature.byFeature({
    features: sample,
    xProperty: 'PC1',
    yProperties: ['PC2'] 
  })
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Score Plot: Distribuzione dei Pixel su PC1 vs PC2',
    hAxis: {title: 'PC1 (' + varianceExplainedList[0].toFixed(1) + '%)'},
    vAxis: {title: 'PC2 (' + varianceExplainedList[1].toFixed(1) + '%)'},
    pointSize: 3,
    legend: {position: 'none'}
  });
print(scorePlot);


// 4.3 Mappe delle Componenti Principali con Palette Unificata
var visParamsPCA = {
  min: -3,
  max: 3,
  palette: ['#2b83ba', '#ffffbf', '#d7191c'] // Blu -> Giallo -> Rosso
};
Map.addLayer(pcImage.select('PC1'), visParamsPCA, 'PC1 - Componente Principale 1');
Map.addLayer(pcImage.select('PC2'), visParamsPCA, 'PC2 - Componente Principale 2', false);
Map.addLayer(pcImage.select('PC3'), visParamsPCA, 'PC3 - Componente Principale 3', false);


//------------------------------------------------------------------------------
// 5. CREAZIONE LEGENDA PER LA MAPPA
//------------------------------------------------------------------------------
var legend = ui.Panel({
  style: { position: 'bottom-left', padding: '8px 15px', width: '270px'}
});
var legendTitle = ui.Label('Legenda PCA', {fontWeight: 'bold', fontSize: '20px', margin: '0 0 4px 0'});
legend.add(legendTitle);
var paletteTitle = ui.Label('Punteggio Componente', {fontWeight: 'bold', fontSize: '16px', margin: '10px 0 6px 8px'});
var legendLabels = ui.Panel({
  widgets: [
    ui.Label('Basso', {margin: '4px 8px'}),
    ui.Label('Medio', {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
    ui.Label('Alto', {margin: '4px 8px'})
  ],
  layout: ui.Panel.Layout.flow('horizontal')
});
var makeColorBar = function(palette) {
  var colorBar = ui.Panel({
    layout: ui.Panel.Layout.flow('horizontal'),
    style: {
      stretch: 'horizontal',
      height: '20px',
      margin: '0 8px'
    }
  });
  palette.forEach(function(color) {
    var colorSegment = ui.Label({
      style: {
        backgroundColor: color,
        stretch: 'horizontal',
        margin: '0',
        height: '20px'
      }
    });
    colorBar.add(colorSegment);
  });
  return colorBar;
};
var palette = ['#2b83ba', '#ffffbf', '#d7191c'];
legend.add(paletteTitle);
legend.add(makeColorBar(palette));
legend.add(legendLabels);
Map.add(legend);


//------------------------------------------------------------------------------
// 6. SERIE TEMPORALI DEGLI INDICI
//------------------------------------------------------------------------------
print('--- Grafici Serie Storiche ---');
print(
  ui.Chart.image.series({
    imageCollection: s1.select(['VV','VH']),
    region: roi, reducer: ee.Reducer.mean(), scale: 30, xProperty: 'datetime'
  })
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Backscatter (dB) VV & VH',
    series: {0:{label:'VV'},1:{label:'VH'}},
    lineWidth:1, pointSize:3,
    hAxis:{title:'Data'}, vAxis:{title:'Backscatter (dB)'}
  })
);
print(
  ui.Chart.image.series({
    imageCollection: s1.select(['RVI', 'RFDI']),
    region: roi, reducer: ee.Reducer.mean(), scale: 30, xProperty: 'datetime'
  })
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Indici Radar: RVI & RFDI',
    series: {0:{label:'RVI'},1:{label:'RFDI'}},
    lineWidth:1, pointSize:3,
    hAxis:{title:'Data'}
  })
);
