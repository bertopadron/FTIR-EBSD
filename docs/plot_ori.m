%%
CS = crystalSymmetry('mmm', [90,50,20])

%%
url1 = 'https://raw.githubusercontent.com/bertopadron/FTIR-EBSD/main/docs/real_orientations.csv';
url2 = 'https://raw.githubusercontent.com/bertopadron/FTIR-EBSD/main/docs/estimated_orientations.csv';
localpath1 = 'C:\Users\Marco\Documents\GitHub\FTIR-EBSD\docs\real_orientations.csv';
localpath2 = 'C:\Users\Marco\Documents\GitHub\FTIR-EBSD\docs\estimated_orientations.csv';

%%
ebsd2 = loadEBSD_generic(url2, ...  % change to localpath for quick tests
                         'CS', CS, ...
                         'ColumnNames', {'phi1', 'PHI', 'phi2', 'X', 'Y'}, ...
                         'Bunge')

ebsd1 = loadEBSD_generic(url1, ...  % change to localpath for quick tests
                         'CS', CS, ...
                         'ColumnNames', {'phi1', 'PHI', 'phi2', 'X', 'Y'}, ...
                         'Bunge')

%%
figure(1)
plot(ebsd1.orientations)
hold on
plot(ebsd2.orientations)

%%
h = Miller({1,0,0}, {0,1,0}, {0,0,1}, ebsd('1').CS);

figure(2)
plotPDF(ebsd1.orientations, h, 'antipodal', 'MarkerSize', 15)
hold on
plotPDF(ebsd2.orientations, h, 'antipodal', 'MarkerSize', 10)

%%
saveFigure('pole_fig_comparison.jpg')

