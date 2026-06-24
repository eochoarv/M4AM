clear all
clc

%% 

Path ='C:\Users\rparham\Documents\Projects\M4AM\CC-OPTIR project\Reference library\20230906 1 to 5 um PP 50-50 EtOH-Omni 2 min sonication iced\Data\Processed\mIRage O-PTIR+Raman\firstlookplots\';


%%

% Import IR data
wavenumbersIR = xlsread('20230906_1to5umPP_2minsonic_iced_30minMPSstg2_allspectra.xlsx',1,'A2:A1029');
spectraExpIR = xlsread('20230906_1to5umPP_2minsonic_iced_30minMPSstg2_allspectra.xlsx',1,'B2:CR1029');

%Import Raman data
wavenumbersRaman = xlsread("20230906_1to5umPP_2minsonic_iced_30minMPSstg2_allspectra.xlsx",3,'A2:A1025');
spectraExpRaman = xlsread('20230906_1to5umPP_2minsonic_iced_30minMPSstg2_allspectra.xlsx',3,'B2:CR1025');

% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR/Raman

%% 

% for i = 1:size(spectraExpIR,2)
%     figure
%     subplot(2,1,1)
%     plot(wavenumbersIR,spectraExpIR(:,i),'Color','#A2142F')
% 
%     xlabel('Wavenumbers (cm^{-1})')
%     ylabel('PTIR Intensity')
%     title(['Particle ',num2str(i)])
% 
%     set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
%     box on
%     
%     subplot(2,1,2)
%     plot(wavenumbersRaman,spectraExpRaman(:,i),'Color','#77AC30')
% 
%     xlabel('Wavenumbers (cm^{-1})')
%     ylabel('Raman Intensity')
% 
%     set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
%     box on
% 
%     FileNamePlot = sprintf('%s%s%d.fig',Path,['Particle ',num2str(i)]);
%     FileNamePlot2 = sprintf('%s%s%d.jpg',Path,['Particle ',num2str(i)]);
%     saveas(gcf,FileNamePlot,'fig')
%     saveas(gcf,FileNamePlot2,'jpg')
% 
%     close all
% end

%% cleaning up

% Note that the last few spectra (i.e. 16-18) have good Raman signal but
% bad IR signal.
keep = [2,3,4,9,11,32,35,39,45,50,51,65,68,71,72,77,90,92,93];

spectraExpIRTrans = transpose(spectraExpIR);
spectraExpRamanTrans = transpose(spectraExpRaman);

count = 1;
for j = 1:length(keep)
    for i = count:size(spectraExpIR,2)
        if i == keep(j)
            count = count+1;
            break
        else
            count = count+1;
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


%% normalize and average spectra
% clear normIR
% clear normRaman
% clear normAvgIR
% clear normAvgRaman
% 
% % normalize IR and Raman spectra
% for i = 1:size(spectraExpIRClean,2)
%     maxIR = max(spectraExpIRClean(:,i));
%     normIR(:,i) = spectraExpIRClean(:,i)/maxIR;
% 
%     maxRaman = max(spectraExpRamanClean(:,i));
%     normRaman(:,i) = spectraExpRamanClean(:,i)/maxRaman;
% end
% 
% % average all of the normalized IR spectra
% for j = 1:size(spectraExpIRClean,1)
%     normAvgIR(j,1) = mean(normIR(j,:));
%     % stdErrorIR(j,1) = std(normAvgIR(j,:))/sqrt(size(spectraExpIRClean,2));
% end
% 
% % average all of the normalized Raman spectra
% for k = 1:size(spectraExpRamanClean,1)
%     normAvgRaman(k,1) = mean(normRaman(k,:));
%     % stdErrorRaman(k,1) = std(normAvgRaman(k,:))/sqrt(size(spectraExpRamanClean,2));
% end


%% preprocess PTIR spectra

% change NaNs to zeroes
for i = 1:size(spectraExpIRClean,2)
    spectraExpIRClean(458:875,i) = 0;
end

% change all negative values to zeroes
for i = 1:size(spectraExpIRClean,2)
    for j = 1:size(spectraExpIRClean,1)
        if spectraExpIRClean(j,i) < 0
            spectraExpIRClean(j,i) = 0;
        end
    end
end

% normalize spectra to their max peaks
for i = 1:size(spectraExpIRClean,2)
    spectrumIRMax = max(spectraExpIRClean(:,i));
    spectraIRNorm(:,i) = spectraExpIRClean(:,i)/spectrumIRMax;
end

% calculate avg and std error of normalized spectra
normAvgIR = mean(spectraIRNorm,2);
normSEIR = std(spectraIRNorm,1,2)./sqrt(size(spectraIRNorm,2));

%% preprocess Raman spectra

% Raman spectra have already had background removed in PTIR Studios

% shift Raman spectra over to correct N2 peak (wavenumbersRaman(578,1) = 2329.2 cm-1)
% note: this will only work if the shift in array elements is no greater
% than +-2 (i.e. +-7 cm-1).
N2standard = 578;

[M,I] = max(spectraExpRamanClean(575:585,:)); % get maximum and index values
N2peaks = I + 574; % adjust index values to match index of wavenumbersRaman

spectraExpRamanClean2 = spectraExpRamanClean; % make new matrix to modify

for i = 1:size(spectraExpRamanClean,2)
    peakValue = N2peaks(1,i);
    while peakValue ~= N2standard
        if peakValue < N2standard
            % make new array with one extra zero at the end
            newArrayLess = [0 ; spectraExpRamanClean2(1:end-1,i)];
            spectraExpRamanClean2(:,i) = newArrayLess;
            peakValue = peakValue+1;
        elseif peakValue > N2standard
            % make new array with one extra zero at the beginning
            newArrayMore = [spectraExpRamanClean2(2:end,i) ; 0];
            spectraExpRamanClean2(:,i) = newArrayMore;
            peakValue = peakValue-1;
        end
    end
end

% normalize spectra to the background peak at ~2442 cm-1
for i = 1:size(spectraExpRamanClean2,2)
    spectrumRamanBkgdPeak = max(spectraExpRamanClean2(593:649,i)); %scanning through 2380-2570 cm-1
    spectraRamanNorm(:,i) = spectraExpRamanClean2(:,i)/spectrumRamanBkgdPeak;
end

% calculate avg and std error of normalized spectra
normAvgRaman = mean(spectraRamanNorm,2);
normSERaman = std(spectraRamanNorm,1,2)./sqrt(size(spectraRamanNorm,2));

%% plot cleaned and normalized PTIR and Raman spectra of unknown particles

figure

% plot all PTIR spectra of unknown particles
subplot(2,1,1)
hold on
% fill([wavenumbersIR, (normAvgIR - normSEIR)], [wavenumbersIR, (normAvgIR + normSEIR)],...
%     [0.6350 0.0780 0.1840],'FaceAlpha',0.2)
plot(wavenumbersIR,normAvgIR,'Color',"#A2142F")
hold off

xlabel('Wavenumbers (cm^{-1})')
ylabel('PTIR Intensity')
title('Norm Avg Reference Spectra for PP')

xlim([500 3500])
ylim([-0.1 1.1])

set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
box on

%plot all Raman spectra of unknown particles
subplot(2,1,2)
hold on
plot(wavenumbersRaman,normAvgRaman,'Color',"#0072BD")
hold off

xlabel('Wavenumbers (cm^{-1})')
ylabel('Raman Intensity')

xlim([500 3500])

set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
box on

%% 


save('PP_50-50_2minsonic_iced_atom_mIRagePTIR_cleandata.mat','wavenumbersIR','normAvgIR','normSEIR')
save('PP_50-50_2minsonic_iced_atom_mIRageRaman_cleandata.mat','wavenumbersRaman','normAvgRaman','normSERaman')

