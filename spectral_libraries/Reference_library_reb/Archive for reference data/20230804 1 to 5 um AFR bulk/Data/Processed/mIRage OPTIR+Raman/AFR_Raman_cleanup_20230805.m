clear all
clc

%% 

Path ='C:\Users\rparham\Documents\Projects\M4AM\CC-OPTIR project\O-PTIR\20230804 1 to 5 um AFR bulk\firstlookplots\';


%% 

wavenumbersRaman = xlsread('20230805_1to5umAFR_bulk_particles.xlsx',1,'L2:L1025');
spectraExp = xlsread('20230805_1to5umAFR_bulk_particles.xlsx',1,'M2:V1025');
% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersRaman

%% 

% for i = 1:size(spectraExp,2)
%     figure
%     plot(wavenumbersRaman,spectraExp(:,i))
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

kill = [];

spectraExpTrans = transpose(spectraExp);

for j = 1:length(kill)
    for i = 1:157
        if i == kill(j)
            spectraExpTrans(i,1) = -9999;
        end
    end
end

spectraExpTrans(spectraExpTrans(:,1) == -9999,:) = [];

spectraExpClean = transpose(spectraExpTrans);
% spectraExpTrans
% each column corresponds to a different particle
% the rows correspond with wavenumbners in wavenumbersIR


%% change negatives to zeros

for i = 1:size(spectraExpClean,2)
    for j = 1:size(spectraExpClean,1)
        if spectraExpClean(j,i) < 0
            spectraExpClean(j,i) = 0;
        end
    end
end


%% normalize spectra

spectrumMax = 0;
spectraNorm = [];

for i = 1:size(spectraExpClean,2)
    spectrumMax = max(spectraExpClean(:,i));
    spectraNorm(:,i) = spectraExpClean(:,i)/spectrumMax;
end

%% calculate average normalized spectrum and standard error

spectrumAvg = zeros(size(spectraNorm(1)));
spectrumStdError = zeros(size(spectraNorm(1)));

for i = 1:size(spectraNorm,1)
    spectrumAvg(i,1) = mean(spectraNorm(i,:));
    spectrumStdError(i,1) = std(spectraNorm(i,:))/sqrt(size(spectraNorm,1));
end

%% 


save('AFR_Raman_cleanup_20230805.mat','spectrumAvg','spectrumStdError')

