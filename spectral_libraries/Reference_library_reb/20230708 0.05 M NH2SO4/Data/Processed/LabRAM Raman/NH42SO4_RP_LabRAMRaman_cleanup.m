% Made on 2024-05-14 by Rebecca Parham (R.P.). Modified XXX by R.P.


clear all
clc

%% create path to save the plots (be sure to make a "firstlookplots" folder)

Path ='C:\Users\rparham\Dropbox (University of Michigan)\LSA-M4AM-Project\CC-OPTIR for MPs\Reference library\20230708 0.05 M NH2SO4\Data\Processed\LabRAM Raman\firstlookplots\';


%% import in all Raman data

%Import Raman data
wavenumbersRaman = xlsread('20240509_50mMNH42SO4_RP_LabRAM_allccdata.xlsx',1,'A2:A2480');
spectraExpRaman = xlsread('20240509_50mMNH42SO4_RP_LabRAM_allccdata.xlsx',1,'B2:BY2480');

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

kill = [18,19,24,37,49,56,63,65];

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

save('50mMNH42SO4_20240514_LabRAMRaman_cleandata.mat','wavenumbersRaman','normAvgRaman','normSERaman')


