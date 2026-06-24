% Made on 2024-05-14 by Rebecca Parham (R.P.). Modified XXX by R.P.

clear all
clc

%% create path to save the plots (be sure to make a "firstlookplots" folder)

Path ='C:\Users\rparham\Dropbox (University of Michigan)\LSA-M4AM-Project\CC-OPTIR for MPs\Reference library\20230708 0.05 M sucrose\Data\Processed\LabRAM Raman\firstlookplots\';


%% import in all Raman data

%Import Raman data
wavenumbersRaman = xlsread('20240509_50mMsucrose_RP_LabRAM_allccdata.xlsx',1,'A2:A2480');
spectraExpRaman = xlsread('20240509_50mMsucrose_RP_LabRAM_allccdata.xlsx',1,'B2:DO2480');

% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR/Raman

%% plot Raman spectra for each particle (use this to make kill array of unwanted spectra)

for i = 1:size(spectraExpRaman,2)
    figure
    plot(wavenumbersRaman,spectraExpRaman(:,i),'Color','#77AC30')

    xlabel('Wavenumbers (cm^{-1})')
    ylabel('Raman Intensity')
    title(['Particle ',num2str(i)])

    set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
    box on

    FileNamePlot = sprintf('%s%s%d.fig',Path,['Particle ',num2str(i)]);
    FileNamePlot2 = sprintf('%s%s%d.jpg',Path,['Particle ',num2str(i)]);
    saveas(gcf,FileNamePlot,'fig')
    saveas(gcf,FileNamePlot2,'jpg')

    close all
end

%% count Raman spectra with too low of S/N to sort

% Removed spectra that seemed like they were contaminated with NaNO3
kill = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,...
    26,27,28,30,31,33,34,35,36,38,39,40,41,42,43,44,46,47,48,49,50,55,...
    56,57,58,59,60,61,62,63,64,65,68,70,73,75,76,77,78,79,80,81,82,86,...
    91,93,94,95,96,97,98,99,100,104,105,106,107,108,110,111,113,114,...
    116,117,118];

spectraExpRamanTrans = transpose(spectraExpRaman);

for j = 1:length(kill)
    for i = 1:size(spectraExpRaman,2)
        if i == kill(j)
            spectraExpRamanTrans(i,1) = -9999;
        end
    end
end

spectraExpRamanTrans(spectraExpRamanTrans(:,1) == -9999,:) = [];

spectraExpRamanClean = transpose(spectraExpRamanTrans);

%% Preprocess Raman spectra

% normalize spectra to the maximum peak (there isn't a background peak)
for i = 1:size(spectraExpRamanClean,2)
    spectrumRamanMax = max(spectraExpRamanClean(:,i)); 
    spectraRamanNorm(:,i) = spectraExpRamanClean(:,i)/spectrumRamanMax;
end

% calculate avg and std error of normalized spectra
normAvgRaman = mean(spectraRamanNorm,2);
normSERaman = std(spectraRamanNorm,1,2)./sqrt(size(spectraRamanNorm,2));

%% plot cleaned and normalized Raman spectra of unknown particles

figure

%plot all Raman spectra of unknown particles
hold on
plot(wavenumbersRaman,normAvgRaman,'Color',"#0072BD")
hold off

xlabel('Wavenumbers (cm^{-1})')
ylabel('Raman Intensity')

xlim([500 3500])

set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
box on
%% Save normalized average Raman spectrum (and SE for spectrum)

save('50mMSucrose_20240514_LabRAMRaman_cleandata.mat','wavenumbersRaman','normAvgRaman','normSERaman')


