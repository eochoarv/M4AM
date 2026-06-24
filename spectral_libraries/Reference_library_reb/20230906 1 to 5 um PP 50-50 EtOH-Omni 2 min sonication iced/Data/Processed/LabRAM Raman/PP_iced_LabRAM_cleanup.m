clear all
clc

%% 

Path ='C:\Users\rparham\Documents\Projects\M4AM\CC-OPTIR project\Reference library\20230906 1 to 5 um PP 50-50 EtOH-Omni 2 min sonication iced\Data\Processed\LabRAM Raman\firstlookplots\';


%%

%Import Raman data
wavenumbersRaman = xlsread("20231116_PP_2minsonic_iced_MPS_AllParticleData.xlsx",1,'A2:A2481');
spectraExpRaman = xlsread('20231116_PP_2minsonic_iced_MPS_AllParticleData.xlsx',1,'B2:I2481');

% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR/Raman

%% 

% number = size(spectraExpRaman,2);
% % number = 1;
% 
% for i = 1:number
%     figure
%     plot(wavenumbersRaman,spectraExpRaman(:,i),'Color','#77AC30')
% 
%     xlabel('Wavenumbers (cm^{-1})')
%     ylabel('Raman Intensity')
%     title(['Particle ',num2str(i)])
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

% only keep particles if Raman spectrum is good
kill = [3,4,8];

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
% spectraExpTrans
% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR/Raman



% %% normalize and average spectra
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
% 
% % calculate SE for IR and Raman normavg spectra
% normSEIR = std(normIR,0,2)./sqrt(size(normIR,2));
% 
% normSERaman = std(normRaman,0,2)./sqrt(size(normRaman,2));


%% preprocess Raman spectra

% Raman spectra have already had background removed in PTIR Studios

% shift Raman spectra over to correct N2 peak (wavenumbersRaman(1161,1) = 2330 cm-1)
% note: this will only work if the shift in array elements is no greater
% than +-2 (i.e. +-7 cm-1).
N2standard = 1161;

[M,I] = max(spectraExpRamanClean(N2standard-10:N2standard+10,:)); % get maximum and index values
N2peaks = I + N2standard - 11; % adjust index values to match index of wavenumbersRaman

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

% normalize spectra to the maximum peak (there isn't a background peak)
for i = 1:size(spectraExpRamanClean2,2)
    spectrumRamanMax = max(spectraExpRamanClean2(:,i)); 
    spectraRamanNorm(:,i) = spectraExpRamanClean2(:,i)/spectrumRamanMax;
end

% calculate avg and std error of normalized spectra
normAvgRaman = mean(spectraRamanNorm,2);
normSERaman = std(spectraRamanNorm,1,2)./sqrt(size(spectraRamanNorm,2));

%% plot cleaned and normalized PTIR and Raman spectra of unknown particles

figure

%plot all Raman spectra of unknown particles
hold on
plot(wavenumbersRaman,normAvgRaman,'Color',"#0072BD")
hold off

xlabel('Wavenumbers (cm^{-1})')
ylabel('Raman Intensity')
title('Norm Avg Reference Raman Spectrum for PP')

xlim([500 3500])

set(gca,'fontsize',12,'linewidth',1,'xdir','reverse')
box on
%% 

save('PP_20231206_LabRAM_cleandata.mat','wavenumbersRaman','normAvgRaman','normSERaman')

