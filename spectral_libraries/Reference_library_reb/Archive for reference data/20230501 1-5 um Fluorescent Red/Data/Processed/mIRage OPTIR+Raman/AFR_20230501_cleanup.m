clear all
clc

%% 

Path ='C:\Users\rparham\Documents\Projects\M4AM\CC-OPTIR project\Reference library\20230501 1-5 um Fluorescent Red\Processed\mIRage OPTIR+Raman\firstlookplots\';


%%

% Import IR data
wavenumbersIR = xlsread('20230501_FluorRed_allccdata.xlsx',1,'A2:A1029');
spectraExpIR = xlsread('20230501_FluorRed_allccdata.xlsx',1,'B2:DU1029');

%Import Raman data
wavenumbersRaman = xlsread("20230501_FluorRed_allccdata.xlsx",2,'A2:A1025');
spectraExpRaman = xlsread('20230501_FluorRed_allccdata.xlsx',2,'B2:DU1025');

% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR/Raman

%% 

for i = 1:size(spectraExpIR,2)
    figure
    subplot(2,1,1)
    plot(wavenumbersIR,spectraExpIR(:,i),'Color','#A2142F')

    xlabel('Wavenumbers (cm^{-1})')
    ylabel('PTIR Intensity')
    title(['Particle ',num2str(i)])

    set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
    box on
    
    subplot(2,1,2)
    plot(wavenumbersRaman,spectraExpRaman(:,i),'Color','#77AC30')

    xlabel('Wavenumbers (cm^{-1})')
    ylabel('Raman Intensity')

    set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
    box on

    FileNamePlot = sprintf('%s%s%d.fig',Path,['Particle ',num2str(i)]);
    FileNamePlot2 = sprintf('%s%s%d.jpg',Path,['Particle ',num2str(i)]);
    saveas(gcf,FileNamePlot,'fig')
    saveas(gcf,FileNamePlot2,'jpg')

    close all
end

%% cleaning up

% all Raman spectra only show fluorescence, so only filtering out
% spectra based on quality of PTIR spectrum
kill = [1,5,9,12,13,15,17,18,20,22,23,24,30,32,33,38,39,45,48,49,50,...
    51,52,70,73,74,75,76,77,78,84,86,88,90,94,101,108,110,113,114,...
    120,124];

spectraExpIRTrans = transpose(spectraExpIR);
spectraExpRamanTrans = transpose(spectraExpRaman);

for j = 1:length(kill)
    for i = 1:size(spectraExpIR,2)
        if i == kill(j)
            spectraExpIRTrans(i,1) = -9999;
            spectraExpRamanTrans(i,1) = -9999;
        end
    end
end

spectraExpIRTrans(spectraExpIRTrans(:,1) == -9999,:) = [];
spectraExpRamanTrans(spectraExpRamanTrans(:,1) == -9999,:) = [];

spectraExpIRClean = transpose(spectraExpIRTrans);
spectraExpRamanClean = transpose(spectraExpRamanTrans);
% spectraExpTrans
% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR/Raman

%% change nans to zeros

% applies to IR spectra only
for i = 1:size(spectraExpIRClean,2)
    spectraExpIRClean(458:875,i) = 0;
end

%% change negatives to zeros

% applies to IR spectra only
for i = 1:size(spectraExpIRClean,2)
    for j = 1:size(spectraExpIRClean,1)
        if spectraExpIRClean(j,i) < 0
            spectraExpIRClean(j,i) = 0;
        end
    end
end

%% normalize and average spectra
clear normIR
clear normRaman
clear normAvgIR
clear normAvgRaman

% normalize IR and Raman spectra
for i = 1:size(spectraExpIRClean,2)
    maxIR = max(spectraExpIRClean(:,i));
    normIR(:,i) = spectraExpIRClean(:,i)/maxIR;

    maxRaman = max(spectraExpRamanClean(:,i));
    normRaman(:,i) = spectraExpRamanClean(:,i)/maxRaman;
end

% average all of the normalized IR spectra
for j = 1:size(spectraExpIRClean,1)
    normAvgIR(j,1) = mean(normIR(j,:));
    % stdErrorIR(j,1) = std(normAvgIR(j,:))/sqrt(size(spectraExpIRClean,2));
end

% average all of the normalized Raman spectra
for k = 1:size(spectraExpRamanClean,1)
    normAvgRaman(k,1) = mean(normRaman(k,:));
    % stdErrorRaman(k,1) = std(normAvgRaman(k,:))/sqrt(size(spectraExpRamanClean,2));
end

% calculate SE for IR and Raman normavg spectra
normSEIR = std(normIR,0,2)./sqrt(size(normIR,2));

normSERaman = std(normRaman,0,2)./sqrt(size(normRaman,2));
%% 


save('AFR_20230501_mIRagePTIR_cleandata.mat','wavenumbersIR','normAvgIR','normSEIR')
save('AFR_20230501_mIRageRaman_cleandata.mat','wavenumbersRaman','normAvgRaman','normSERaman')

